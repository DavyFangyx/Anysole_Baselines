"""Canonical Step2Motion exports keep source frame ids and layout provenance."""
from pathlib import Path
import numpy as np


def export_canonical_motion(path, dataset, checkpoint_dir, pose_layout):
    from Baselines.utils.gt_loading import shared_session_frames
    from Baselines.utils.prediction_grid import bvh_prediction_on_grid
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
    points, valid, names, rotations = bvh_prediction_on_grid(path, dict(session_id=sid, n_frames=n))
    folder = path.parent / 'eval_motion'
    folder.mkdir(parents=True, exist_ok=True)
    # Separate clip names prevent an internal gap from overwriting another clip.
    output = folder / (dataset.clip_name(0) + '.npz')
    np.savez_compressed(output, joint_xyz_world=points, valid_mask=valid,
                        joint_names=np.asarray(names), joint_rotations=rotations,
                        frame_indices=np.arange(n, dtype=np.int64),
                        motion_protocol=np.asarray('bvh23'), model_units=np.asarray('m'),
                        coordinate_system=np.asarray('world_z_up'),
                        joint_coordinate_system=np.asarray('world_z_up'),
                        rotation_coordinate_system=np.asarray('bvh_native_y_up'),
                        session_id=np.asarray(sid), target_fps=np.asarray(dataset.sample_rate),
                        pose_layout=np.asarray(pose_layout),
                        source_native_output=np.asarray(str(path)),
                        source_checkpoint=np.asarray(str(checkpoint_dir)),
                        provenance_source_type=np.asarray('model_prediction'),
                        time_axis_source=np.asarray(facts['source_path']))
    return output
