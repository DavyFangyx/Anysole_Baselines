import copy
import math
import numpy as np
import os
import scipy
import torch
import torch.nn as nn
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # noqa: isort

from lib.utils.depth_utils import depth2PointCloud  # noqa: E402
from lib.utils.render_utils import modelRender  # noqa: E402

# kd-tree query parallelism: scipy's workers=_KD_WORKERS spawns os.cpu_count() threads
# per fitting process, so N concurrent processes thrash N*96 threads on the
# CPU. run_full_mmvp.py caps this per child (cores / concurrency) via
# PRESSURE_KDTREE_WORKERS; the default below stays safe for a solo run.
_KD_WORKERS = int(os.environ.get("PRESSURE_KDTREE_WORKERS",
                                 min(os.cpu_count(), 16)))

# PRESSURE_ICP_DEVICE=cuda moves the 550k-point correspondence pipeline
# (kNN, normals, masks, gathers) onto the GPU; default 'cpu' keeps the
# original numpy/cKDTree path bit-for-bit. GPU results differ at ~1e-6
# (float32 summation order + topk tie-breaking), A/B before adopting.
_ICP_DEVICE = os.environ.get("PRESSURE_ICP_DEVICE", "cpu")
# distance-matrix row budget per cdist call in the GPU kNN helpers
_ICP_CHUNK = int(os.environ.get("PRESSURE_ICP_CHUNK", "32768"))
_GPU_NORMALS = os.environ.get("PRESSURE_GPU_NORMALS", "0") == "1"
_GPU_VISIBILITY_EXACT = os.environ.get(
    "PRESSURE_GPU_VISIBILITY_EXACT", "0") == "1"


def _gpu_knn(query, ref, k):
    """Exact kNN on GPU, chunked over query rows.

    Returns (dists, indices), each [nq, k], distances sorted ascending.
    Tie order may differ from scipy cKDTree for exactly equal distances.
    Distances use the |x|^2 - 2 x.y + |y|^2 expansion instead of cdist,
    which avoids cdist's [n, m, 3] intermediate and keeps GPU memory
    bounded by the chunk's 2D distance matrix.
    """
    nq = query.shape[0]
    out_d = torch.empty((nq, k), dtype=query.dtype, device=query.device)
    out_i = torch.empty((nq, k), dtype=torch.long, device=query.device)
    ref_norm = (ref * ref).sum(1)
    for s in range(0, nq, _ICP_CHUNK):
        q = query[s:s + _ICP_CHUNK]
        d = (q * q).sum(1, keepdim=True) - 2 * q @ ref.T + ref_norm[None, :]
        v, i = torch.topk(-d, k=k, dim=1)
        out_d[s:s + _ICP_CHUNK] = -v
        out_i[s:s + _ICP_CHUNK] = i
    return out_d, out_i


def _gpu_knn_small_query_big_ref(query, ref, k):
    """Same as _gpu_knn but chunks over *ref* rows (few queries, big ref)."""
    q_norm = (query * query).sum(1, keepdim=True)
    parts_d, parts_i = [], []
    for s in range(0, ref.shape[0], _ICP_CHUNK):
        r = ref[s:s + _ICP_CHUNK]
        d = q_norm - 2 * query @ r.T + (r * r).sum(1)[None, :]
        v, i = torch.topk(-d, k=k, dim=1)
        parts_d.append(-v)
        parts_i.append(i + s)
    merged_d = torch.cat(parts_d, dim=1)
    merged_i = torch.cat(parts_i, dim=1)
    v, idx = torch.topk(-merged_d, k=k, dim=1)
    return -v, torch.gather(merged_i, 1, idx)


class DepthTerm(nn.Module):

    def __init__(self,
                 essential_root=None,
                 cam_intr=None,
                 img_W=640,
                 img_H=576,
                 faces=None,
                 save_obj=False,
                 dtype=np.float32,
                 device='cpu'):
        super(DepthTerm, self).__init__()

        self.cam_intr = cam_intr.cpu().numpy()  # fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.dtype = dtype
        self.device = device
        hand_ids = np.loadtxt(f'{essential_root}/hand_ids.txt').astype(
            np.int32)
        self.valid_verts = np.ones(6890)
        self.valid_verts[hand_ids] = 0

        self.renderer = modelRender(cam_intr, img_W, img_H)

        # set() membership makes the visible-vertex filter O(n) instead of O(n^2)
        self.foot_ids_surfaces = set(
            np.load(
                f'{essential_root}/foot_related/foot_ids_surfaces.npy'
            ).tolist())

        self.model_faces = faces
        self.faces_t = torch.as_tensor(
            np.asarray(faces), dtype=torch.long)
        self.gpu_normals = _GPU_NORMALS
        self.gpu_visibility_exact = _GPU_VISIBILITY_EXACT

        # GPU correspondence pipeline (PRESSURE_ICP_DEVICE=cuda) and its
        # per-frame static-depth cache. The cache is keyed by data_ptr and
        # lives only as long as this DepthTerm instance, which is rebuilt
        # for every frame, so no cross-frame staleness is possible.
        self.icp_device = 'cuda' if (
            _ICP_DEVICE == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self._depth_key = None
        self._depth_vmap_t = None
        self._depth_nmap_t = None

        self.save_obj = save_obj
        self.debug_dir = os.path.join(os.getcwd(), 'debug')
        if self.save_obj:
            os.makedirs(self.debug_dir, exist_ok=True)

    def reset_frame_cache(self):
        """Drop frame-local depth tensors when this term is reused."""
        self._depth_key = None
        self._depth_vmap_t = None
        self._depth_nmap_t = None

    def vertex_normals_torch(self, vertices):
        """Angle-weighted vertex normals, matching trimesh's convention.

        Correspondence selection is non-differentiable in the original path,
        so callers pass detached vertices.  The implementation is device
        agnostic and becomes a GPU implementation when ``vertices`` is CUDA.
        """
        faces = self.faces_t.to(device=vertices.device)
        triangles = vertices[faces]
        e01 = triangles[:, 1] - triangles[:, 0]
        e02 = triangles[:, 2] - triangles[:, 0]
        e10 = triangles[:, 0] - triangles[:, 1]
        e12 = triangles[:, 2] - triangles[:, 1]
        e20 = triangles[:, 0] - triangles[:, 2]
        e21 = triangles[:, 1] - triangles[:, 2]

        face_cross = torch.cross(e01, e02, dim=1)
        face_norm = torch.nn.functional.normalize(face_cross, dim=1)

        def corner_angle(a, b):
            a = torch.nn.functional.normalize(a, dim=1)
            b = torch.nn.functional.normalize(b, dim=1)
            cosine = (a * b).sum(dim=1).clamp(-1.0, 1.0)
            return torch.acos(cosine)

        angles = torch.stack((
            corner_angle(e01, e02),
            corner_angle(e10, e12),
            corner_angle(e20, e21),
        ), dim=1)
        weighted = face_norm[:, None, :] * angles[:, :, None]
        result = torch.zeros(
            (vertices.shape[0], 3), dtype=vertices.dtype,
            device=vertices.device)
        for corner in range(3):
            result.index_add_(0, faces[:, corner], weighted[:, corner])
        return torch.nn.functional.normalize(result, dim=1)

    def findLiveVisibileVerticesIndex(self, mesh, near_size=4, th=0.005):
        _mesh = copy.deepcopy(mesh)
        color_render, depth = self.renderer.render(_mesh)

        # cv2.imwrite('debug/new_framework/test.png',color_render)
        # _mesh.export('debug/new_framework/mesh_.obj')
        # import pdb;pdb.set_trace()

        # mesh.export('debug/mesh.obj')
        point_cloud = depth2PointCloud(depth, self.cam_intr[0],
                                       self.cam_intr[1], self.cam_intr[2],
                                       self.cam_intr[3])
        kdtree = scipy.spatial.cKDTree(point_cloud.reshape([-1, 3]))
        dists, indices = kdtree.query(_mesh.vertices, k=near_size, workers=_KD_WORKERS)
        # cKDTree.query returns k nearest sorted ascending, so the first
        # column already is the min; avoids a full (n,k) reduction.
        min_dists = dists[:, 0]
        flame_visible_idx = np.where((min_dists < th)
                                     & (self.valid_verts > 0.5))[0]

        return flame_visible_idx

    def findLiveVisibleVerticesIndexGPUExact(self, mesh, vertices,
                                             near_size=4, th=0.005):
        """GPU equivalent of findLiveVisibileVerticesIndex().

        This intentionally preserves the original operation order at the
        semantic level: the same pyrender depth buffer is converted to the
        same full 3-D point cloud, then each SMPL vertex queries k=4 nearest
        points using squared Euclidean distances on GPU.  Only the cKDTree
        implementation is replaced by chunked torch top-k.
        """
        _, depth = self.renderer.render(copy.deepcopy(mesh))
        point_cloud = depth2PointCloud(
            depth, self.cam_intr[0], self.cam_intr[1],
            self.cam_intr[2], self.cam_intr[3]).reshape(-1, 3)
        # scipy cKDTree evaluates the float32 point cloud with double-
        # precision distance accumulation.  Use float64 here to avoid
        # visibility threshold/tie changes from float32 top-k arithmetic.
        visibility_dtype = torch.float64
        depth_points_t = torch.from_numpy(point_cloud).to(
            device=vertices.device, dtype=visibility_dtype)
        distances_sq, _ = _gpu_knn_small_query_big_ref(
            vertices.detach().to(dtype=visibility_dtype), depth_points_t,
            near_size)
        visible = torch.sqrt(torch.clamp(distances_sq[:, 0], min=0)) < th
        valid_verts = torch.as_tensor(
            self.valid_verts > 0.5, device=vertices.device, dtype=torch.bool)
        return torch.nonzero(visible & valid_verts,
                             as_tuple=False)[:, 0]

    def findCorrs(self,
                  depth_vmap=None,
                  depth_nmap=None,
                  live_verts=None,
                  icp_near_size=32,
                  icp_theta_thresh=np.pi / 12,
                  icp_dist_thresh=0.05):
        # TODO: run ICP on cuda
        depth_vmap = depth_vmap.cpu().numpy()
        depth_nmap = depth_nmap.cpu().numpy()
        live_verts = live_verts.detach().cpu().numpy()

        live_mesh = trimesh.Trimesh(
            vertices=live_verts, faces=self.model_faces, process=False)

        # live_mesh.export('debug/new_framework/smpl_vertices.obj')
        # trimesh.Trimesh(vertices=depth_vmap).export('debug/new_framework/gt_depth_vmap.obj')
        # import pdb;pdb.set_trace()

        live_normals = live_mesh.vertex_normals
        smpl_visible_idx_w_foot = self.findLiveVisibileVerticesIndex(live_mesh)
        smpl_visible_idx = [
            idx for idx in smpl_visible_idx_w_foot
            if idx not in self.foot_ids_surfaces
        ]

        verts_src = live_verts[smpl_visible_idx, :]
        normal_src = live_normals[smpl_visible_idx, :]
        # trimesh.Trimesh(vertices=verts_src).export('debug/new_framework/smpl_visible.obj')

        # find smpl verts currs to depth, each smpl v corres to one depth v
        kdtree_depth = scipy.spatial.cKDTree(depth_vmap)
        dists_smpl2depth, indices_smpl2depth = kdtree_depth.query(
            verts_src, k=icp_near_size, workers=_KD_WORKERS)

        tar_normals_smpl2depth = depth_nmap[indices_smpl2depth.reshape(
            -1)].reshape(-1, icp_near_size, 3)

        # einsum('ijk,ik->ij') is a batched 3-vector dot; BLAS matmul runs the
        # same math ~3x faster than einsum and ~10x faster than a broadcast
        # mul+sum (which materializes a full [N,32,3] temporary).
        cosine_smpl2depth = np.matmul(
            tar_normals_smpl2depth, normal_src[:, :, None])[:, :, 0]
        valid_smpl2depth = (dists_smpl2depth < icp_dist_thresh) & (
            cosine_smpl2depth > math.cos(icp_theta_thresh))
        valid_indices_smpl2depth = np.argmax(valid_smpl2depth, axis=1)
        indices_corr_smpl2depth = np.take_along_axis(
            indices_smpl2depth, valid_indices_smpl2depth[:, None], axis=1)[:, 0]

        # find depth verts currs to smpl, each depth v corres to one smpl v
        kdtree_depth2smpl = scipy.spatial.cKDTree(verts_src)
        dists_depth2smpl, indices_depth2smpl = kdtree_depth2smpl.query(
            depth_vmap, k=icp_near_size, workers=_KD_WORKERS)

        tar_normals_depth2smpl = normal_src[indices_depth2smpl.reshape(
            -1)].reshape(-1, icp_near_size, 3)
        cosine_depth2smpl = np.matmul(
            tar_normals_depth2smpl, depth_nmap[:, :, None])[:, :, 0]
        valid_depth2smpl = (dists_depth2smpl < icp_dist_thresh) & (
            cosine_depth2smpl > math.cos(icp_theta_thresh))
        valid_indices_depth2smpl = np.argmax(valid_depth2smpl, axis=1)
        indices_corr_depth2smpl = np.take_along_axis(
            indices_depth2smpl, valid_indices_depth2smpl[:, None], axis=1)[:, 0]

        # save
        if self.save_obj:
            tar_verts = depth_vmap[indices_corr_smpl2depth]
            with open(os.path.join(self.debug_dir, 'corrs_smpl2depth.obj'), 'w') as fp:
                for vi in range(verts_src.shape[0]):
                    fp.write(
                        'v %f %f %f\n' %
                        (verts_src[vi, 0], verts_src[vi, 1], verts_src[vi, 2]))
                    fp.write(
                        'v %f %f %f\n' %
                        (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
                for li in range(verts_src.shape[0]):
                    fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))

            tar_verts = verts_src[indices_corr_depth2smpl]
            with open(os.path.join(self.debug_dir, 'corrs_depth2smpl.obj'), 'w+') as fp:
                for vi in range(depth_vmap.shape[0]):
                    fp.write('v %f %f %f\n' %
                             (depth_vmap[vi, 0], depth_vmap[vi, 1],
                              depth_vmap[vi, 2]))
                    fp.write(
                        'v %f %f %f\n' %
                        (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
                for li in range(depth_vmap.shape[0]):
                    fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))
        return smpl_visible_idx, indices_corr_smpl2depth, \
            indices_corr_depth2smpl

    def findCorrsGPU(self,
                     depth_vmap_t=None,
                     depth_nmap_t=None,
                     live_verts_t=None,
                     icp_near_size=32,
                     icp_theta_thresh=np.pi / 12,
                     icp_dist_thresh=0.05):
        """GPU mirror of findCorrs().

        Same correspondences and same masks, but the kNN queries, cosine
        tests, argmax picks and gathers all run in torch on the GPU.
        The visibility render stays on CPU (pyrender/EGL) and only needs the
        6890-vertex mesh, so the transfer is tiny.
        """
        live_mesh = trimesh.Trimesh(
            vertices=live_verts_t.detach().cpu().numpy(),
            faces=self.model_faces,
            process=False)
        if self.gpu_normals:
            with torch.no_grad():
                live_normals_t = self.vertex_normals_torch(
                    live_verts_t.detach())
            live_normals = None
        else:
            live_normals_t = None
            live_normals = live_mesh.vertex_normals
        if self.gpu_visibility_exact:
            smpl_visible_idx_w_foot_t = self.findLiveVisibleVerticesIndexGPUExact(
                live_mesh, live_verts_t)
            foot_ids_t = torch.as_tensor(
                list(self.foot_ids_surfaces), dtype=torch.long,
                device=live_verts_t.device)
            vis_idx_t = smpl_visible_idx_w_foot_t[
                ~torch.isin(smpl_visible_idx_w_foot_t, foot_ids_t)]
            smpl_visible_idx = vis_idx_t.detach().cpu().tolist()
        else:
            smpl_visible_idx_w_foot = self.findLiveVisibileVerticesIndex(
                live_mesh)
            smpl_visible_idx = [
                idx for idx in smpl_visible_idx_w_foot
                if idx not in self.foot_ids_surfaces
            ]
            vis_idx_t = torch.tensor(smpl_visible_idx,
                                     dtype=torch.long,
                                     device=live_verts_t.device)
        verts_src_t = live_verts_t[vis_idx_t]
        if live_normals_t is not None:
            normal_src_t = live_normals_t[vis_idx_t]
        else:
            normal_src_t = torch.from_numpy(live_normals[smpl_visible_idx]).to(
                device=live_verts_t.device, dtype=live_verts_t.dtype)
        cos_th = math.cos(icp_theta_thresh)

        # find smpl verts currs to depth, each smpl v corres to one depth v
        dists_s2d, indices_s2d = _gpu_knn_small_query_big_ref(
            verts_src_t, depth_vmap_t, icp_near_size)
        tar_normals_s2d = depth_nmap_t[indices_s2d]
        cosine_s2d = (tar_normals_s2d *
                      normal_src_t[:, None, :]).sum(-1)
        valid_s2d = (dists_s2d < icp_dist_thresh) & (cosine_s2d > cos_th)
        first_valid_s2d = valid_s2d.int().argmax(dim=1)
        indices_corr_s2d = torch.gather(indices_s2d, 1,
                                        first_valid_s2d[:, None])[:, 0]

        # find depth verts currs to smpl, each depth v corres to one smpl v
        dists_d2s, indices_d2s = _gpu_knn(depth_vmap_t, verts_src_t,
                                          icp_near_size)
        tar_normals_d2s = normal_src_t[indices_d2s]
        cosine_d2s = (tar_normals_d2s *
                      depth_nmap_t[:, None, :]).sum(-1)
        valid_d2s = (dists_d2s < icp_dist_thresh) & (cosine_d2s > cos_th)
        first_valid_d2s = valid_d2s.int().argmax(dim=1)
        indices_corr_d2s = torch.gather(indices_d2s, 1,
                                        first_valid_d2s[:, None])[:, 0]

        return vis_idx_t, indices_corr_s2d, indices_corr_d2s

    def forward(self, depth_vmap=None, depth_nmap=None, live_verts=None):
        """_summary_

        Args:
            depth_vmap (torch.tensor, optional): depth cloud. Defaults to None.
            depth_nmap (torch.tensor, optional): depth cloud norm.
                Defaults to None.
            live_verts (torch.Size([6890, 3]), optional):
                smpl surface vertices. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.icp_device == 'cuda':
            if self._depth_key != depth_vmap.data_ptr():
                self._depth_vmap_t = depth_vmap.detach()
                self._depth_nmap_t = depth_nmap.detach()
                self._depth_key = depth_vmap.data_ptr()
            smpl_ids_vis, depth_ids, smpl_ids = self.findCorrsGPU(
                depth_vmap_t=self._depth_vmap_t,
                depth_nmap_t=self._depth_nmap_t,
                live_verts_t=live_verts)

            src_verts_smpl2depth = live_verts[smpl_ids_vis, :]
            tar_verts_smpl2depth = self._depth_vmap_t[depth_ids, :]
            src_verts_depth2smpl = self._depth_vmap_t
            tar_verts_depth2smpl = live_verts[smpl_ids_vis, :][smpl_ids, :]

            delta_smpl2depth = src_verts_smpl2depth - tar_verts_smpl2depth
            dist_smpl2depth = torch.norm(delta_smpl2depth, dim=-1)
            delta_depth2smpl = src_verts_depth2smpl - tar_verts_depth2smpl
            dist_depth2smpl = torch.norm(delta_depth2smpl, dim=-1)

            depth_loss = torch.mean(
                dist_smpl2depth, dim=0) + torch.mean(
                    dist_depth2smpl, dim=0)

            return depth_loss

        smpl_ids_vis, depth_ids, smpl_ids = self.findCorrs(
            depth_vmap=depth_vmap,
            depth_nmap=depth_nmap,
            live_verts=live_verts)

        # calculate smpl2depth corres loss
        smpl_ids_vis = torch.tensor(smpl_ids_vis, device=self.device).long()
        # depth_ids = torch.tensor(depth_ids, device= self.device).long()
        src_verts_smpl2depth = live_verts[smpl_ids_vis, :]
        tar_verts_smpl2depth = depth_vmap[depth_ids, :]
        # calculate depth2smpl corres loss
        src_verts_depth2smpl = depth_vmap
        tar_verts_depth2smpl = live_verts[smpl_ids_vis, :][smpl_ids, :]

        delta_smpl2depth = src_verts_smpl2depth - tar_verts_smpl2depth
        dist_smpl2depth = torch.norm(delta_smpl2depth, dim=-1)
        delta_depth2smpl = src_verts_depth2smpl - tar_verts_depth2smpl
        dist_depth2smpl = torch.norm(delta_depth2smpl, dim=-1)
        # import pdb;pdb.set_trace()

        depth_loss = torch.mean(
            dist_smpl2depth, dim=0) + torch.mean(
                dist_depth2smpl, dim=0)

        return depth_loss
