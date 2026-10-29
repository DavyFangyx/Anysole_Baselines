"""Canonical Step2Motion exports keep source frame ids and layout provenance."""
from pathlib import Path
import numpy as np


def export_canonical_motion(path, dataset, checkpoint_dir, pose_layout,
                            world=None, rotations=None):
    from Baselines.utils.gt_loading import shared_session_frames
    path = Path(path)
    sid = str(dataset.session_ids[0])
    facts = shared_session_frames(dict(session_id=sid, camera='cam3'))
    n = len(facts['frame_id'])
    np.savez_compressed(path.with_suffix('.frames.npz'),
                        frame_indices=dataset.inference_frame_ids,
                        valid_mask=dataset.inference_valid,
                        session_id=np.asarray(sid),
                        pose_layout=np.asarray(pose_layout),
                        target_fps=np.asarray(dataset.sample_rate))
    ids = np.asarray(dataset.inference_frame_ids, dtype=np.int64)
    keep = np.asarray(dataset.inference_valid, dtype=bool)
    if world is None or rotations is None:
        from Baselines.utils.prediction_grid import bvh_prediction_on_grid
        world, keep, names, rotations = bvh_prediction_on_grid(
            path, dict(session_id=sid, n_frames=n))
        full_world = world
        full_rotations = rotations
    else:
        names = tuple(str(v) for v in np.asarray(dataset.joint_names))
        if world.shape[0] != len(ids) or world.shape[1] != len(names):
            raise ValueError(
                f"pred world {world.shape} does not match inference frames "
                f"{len(ids)} x {len(names)} joints"
            )
        if not np.issubdtype(ids.dtype, np.integer) or np.any(np.diff(ids) <= 0):
            raise ValueError("Invalid/duplicated frame ids in generated BVH sidecar")
        # Joints land on the canonical grid in the z-up display frame;
        # rotations stay SMPL-native y-up (the smpl24 evaluation contract).
        display = np.stack([world[..., 0], -world[..., 2], world[..., 1]], axis=-1)
        full_world = np.zeros((n, len(names), 3), dtype=np.float32)
        full_rotations = np.zeros((n, len(names), 3, 3), dtype=np.float32)
        full_world[ids] = display.astype(np.float32)
        full_rotations[ids] = rotations.astype(np.float32)
        mask = np.zeros(n, dtype=bool)
        mask[ids] = keep
        keep = mask
    folder = path.parent / 'eval_motion'
    folder.mkdir(parents=True, exist_ok=True)
    # Separate clip names prevent an internal gap from overwriting another clip.
    output = folder / (dataset.clip_name(0) + '.npz')
    protocol = 'smpl24' if list(names) and str(names[0]) == 'pelvis' else 'bvh23'
    np.savez_compressed(output, joint_xyz_world=full_world, valid_mask=keep,
                        joint_names=np.asarray(names), joint_rotations=full_rotations,
                        frame_indices=np.arange(n, dtype=np.int64),
                        motion_protocol=np.asarray(protocol), model_units=np.asarray('m'),
                        coordinate_system=np.asarray('world_z_up'),
                        joint_coordinate_system=np.asarray('world_z_up'),
                        rotation_coordinate_system=np.asarray('smpl_native_y_up'),
                        session_id=np.asarray(sid), target_fps=np.asarray(dataset.sample_rate),
                        pose_layout=np.asarray(pose_layout),
                        source_native_output=np.asarray(str(path)),
                        source_checkpoint=np.asarray(str(checkpoint_dir)),
                        provenance_source_type=np.asarray('model_prediction'),
                        time_axis_source=np.asarray(facts['source_path']))
    return output
