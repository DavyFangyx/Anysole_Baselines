"""Exercise actual export functions with synthetic assets and independent FK."""
from pathlib import Path
import importlib.util,sys,tempfile,types,json
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];HERE=Path(tempfile.gettempdir())
sys.path.insert(0,str(ROOT))
from anysole.utils.geometry import fk_local_np

class FakeSMPL:
    def __init__(self,*a,**kw):pass
    def cpu(self):return self
    def setPose(self,**kw):pass
    def update_shape(self):pass
    def init_plane(self):pass
    def update_pose(self):return types.SimpleNamespace(vertices=torch.zeros(1,6890,3))
    def __call__(self,**kw):
        n=len(kw['transl'])
        return types.SimpleNamespace(vertices=torch.arange(n,dtype=torch.float32)[:,None,None].expand(n,6890,3).clone())

def exercise(source,kind,frame_ids=None):
    spec=importlib.util.spec_from_file_location('export_under_test',source)
    m=importlib.util.module_from_spec(spec)
    if kind=='vp_side':
        m.__file__=str(ROOT/'AnysoleWorkspace/tool/adapters/mmvp_series/posetransopt/export_v2m.py')
        exec(compile(Path(source).read_text(),str(source),'exec'),m.__dict__)
    else:spec.loader.exec_module(m)
    rng=np.random.default_rng(20261022);n=4
    ids=np.asarray(frame_ids if frame_ids is not None else np.arange(2,n+2),dtype=np.int64)
    aa=rng.normal(0,.3,(n,24,3));aa[0]=0
    local=Rotation.from_rotvec(aa.reshape(-1,3)).as_matrix().reshape(n,24,3,3)
    root=rng.normal(0,.1,(n,3))
    offsets=rng.normal(0,.04,(24,3));offsets[0]=0
    parents=np.arange(24)-1
    A=Rotation.from_rotvec([.1,-.2,.15]).as_matrix();b=np.array([.12,.23,.34])
    transformed=local.copy();transformed[:,0]=A@local[:,0]
    pos,_=fk_local_np(transformed,offsets,parents)
    expected=pos+(root@A.T+b)[:,None,:]
    expected=m.smpl_yup_to_display(expected)
    captured={}
    def capture(output,joints,valid,*args,**kwargs):
        captured.update(joints=joints.copy(),valid=valid.copy(),vertices=kwargs.get('vertices'),poses=kwargs.get('poses'))
    fake=types.ModuleType('lib.core.smpl_mmvp');fake.SMPL_MMVP=FakeSMPL
    with tempfile.TemporaryDirectory(prefix='export_rotation_test_',dir=str(HERE)) as td:
        t=Path(td);m.WORKSPACE=t;m.ROOT=ROOT
        m.resolve_uri=lambda x,**kw:Path(x)
        m.session_parts=lambda row:('date','subject','SYNTH') if kind=='vp_side' else ('date','subject')
        m.WORK_ROOT=t/'work/VP-MoCap/v1/pose_optimization'
        m.OUTPUT_ROOT=t/'side_out'
        m.shared_valid=lambda row:np.ones(int(row['n_frames']),bool)
        m.valid_frames=lambda row,n:np.ones(n,bool)
        m.write_archive=capture
        paths=[]
        for i in range(n):
            p=t/f'smpl_{i:06d}.npz'
            np.savez(p,shape=np.zeros(10),body_pose=aa[i,1:].reshape(1,69),global_rot=aa[i,0],transl=root[i])
            paths.append(p)
        m.pressure_frame_paths=lambda *args:paths
        native=t/'work/VP-MoCap/v1/pose_optimization/date/subject/SYNTH/opt_result.pth'
        native.parent.mkdir(parents=True)
        torch.save(dict(pose=torch.tensor(local,dtype=torch.float32),beta=torch.zeros(n,10),trans=torch.tensor(root,dtype=torch.float32),frame_ids=torch.from_numpy(ids)),native)
        transforms={'A':A,'b':b,'artifact_path':str(t/'calibration.json'),'source':str(t/'calibration.json')}
        gp=types.ModuleType('AnysoleWorkspace.tool.adapters.mmvp_series.pressure_tookit.native_to_mocap')
        gp.load_artifact=lambda date:transforms
        vp=types.ModuleType('AnysoleWorkspace.tool.adapters.mmvp_series.posetransopt.native_to_mocap')
        vp.load_artifact=lambda date:transforms
        gio=types.ModuleType('anysole.data.smpl_io')
        gio.load_smpl=lambda path:{'offsets_m':offsets,'parents':parents}
        with patch.dict(sys.modules,{'lib.core.smpl_mmvp':fake,gp.__name__:gp,vp.__name__:vp,gio.__name__:gio}), \
             patch('smplx.create',return_value=FakeSMPL()):
            row={'session_id':'SYNTH','n_frames':n if kind=='pressure' else n+4,'smpl_path':str(t/'target.npz')}
            args=types.SimpleNamespace(female='',pressure_root=str(t))
            if kind=='pressure':m.export_pressure(row,args,t/'out.npz');actual=captured['joints']
            elif kind=='vp':
                m.export_vp_mocap(row,args,t/'out.npz')
                actual=captured['joints'][ids]
                available=np.zeros(n+4,bool);available[ids]=True
                np.testing.assert_array_equal(captured['valid'],available)
                np.testing.assert_array_equal(captured['joints'][~available],0)
                np.testing.assert_array_equal(captured['poses'][~available],0)
                np.testing.assert_allclose(captured['poses'][ids,3:],aa[:,1:].reshape(n,69),atol=2e-7)
                native_vertices=np.broadcast_to(np.arange(n)[:,None,None],(n,6890,3))
                np.testing.assert_array_equal(captured['vertices'][ids],m.smpl_yup_to_display(native_vertices))
                np.testing.assert_array_equal(captured['vertices'][~available],0)
            else:
                m.export_session(row,True)
                with np.load(m.OUTPUT_ROOT/'SYNTH.npz') as archive:
                    actual=archive['joint_xyz_world'][2:2+n]
        return float(np.max(abs(actual-expected)))

def test_rotation_encoder_roundtrip():
    from anysole.utils.geometry import rotmat_to_6d_np, rot6d_to_rotmat_np
    r=Rotation.from_rotvec(np.array([[0.,0.,0.],[.2,-.4,.6]])).as_matrix()
    np.testing.assert_allclose(rot6d_to_rotmat_np(rotmat_to_6d_np(r)),r,atol=2e-7)

def test_pressure_export_preserves_body_rotations():
    assert exercise(ROOT/'AnysoleWorkspace/tool/export_baseline_motion.py','pressure')<1e-6

def test_vp_export_uses_column_stacked_6d():
    assert exercise(ROOT/'AnysoleWorkspace/tool/export_baseline_motion.py','vp')<1e-6

def test_vp_side_export_uses_column_stacked_6d():
    assert exercise(ROOT/'AnysoleWorkspace/tool/adapters/mmvp_series/posetransopt/export_v2m.py','vp_side')<1e-6


def test_vp_export_places_gapped_frame_ids():
    assert exercise(ROOT/'AnysoleWorkspace/tool/export_baseline_motion.py','vp',[1,4,6,7])<1e-6


def test_vp_export_rejects_duplicate_or_out_of_range_frame_ids():
    source=ROOT/'AnysoleWorkspace/tool/export_baseline_motion.py'
    for ids in ([1,4,4,7],[-1,2,3,4],[1,2,3,8]):
        try:
            exercise(source,'vp',ids)
        except ValueError as exc:
            assert 'frame_ids' in str(exc)
        else:
            raise AssertionError('invalid frame ids accepted: '+str(ids))
