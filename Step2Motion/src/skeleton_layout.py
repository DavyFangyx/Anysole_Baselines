"""Anatomical joint groups independent of the serialized skeleton order."""
from bvh_export import DEFAULT_JOINT_NAMES

GAIT_JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
]

# Native SMPL-24 skeleton (npz joint order), the gait adapter v2 layout.
SMPL24_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
]
LEFT_LEG = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
RIGHT_LEG = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']


def joint_names(n_joints, names=None):
    if names is None:
        if n_joints == len(GAIT_JOINT_NAMES):
            names = GAIT_JOINT_NAMES
        elif n_joints == len(SMPL24_JOINT_NAMES):
            names = SMPL24_JOINT_NAMES
        elif n_joints == len(DEFAULT_JOINT_NAMES):
            names = DEFAULT_JOINT_NAMES
        else:
            raise ValueError('Unknown skeleton: supply skeleton_joint_names explicitly')
    names = list(names)
    if len(names) != n_joints or len(set(names)) != n_joints or names[0] not in ('Hips', 'pelvis'):
        raise ValueError('Skeleton names must be unique, match the pose width, and start with Hips/pelvis')
    return names


def joint_groups(n_joints, names=None):
    names = joint_names(n_joints, names)
    def indices(group):
        missing = set(group) - set(names)
        if missing:
            raise ValueError('Skeleton missing anatomical joints: ' + repr(sorted(missing)))
        return [names.index(name) - 1 for name in group]
    left, right = indices(LEFT_LEG), indices(RIGHT_LEG)
    body = [i for i in range(n_joints - 1) if i not in left + right]
    return left, right, body


def pose_channel_order(n_joints, names=None):
    left, right, body = joint_groups(n_joints, names)
    return [3 * j + axis for j in left + right + body for axis in range(3)]


def metric_joint_indices(n_joints, names=None):
    names = joint_names(n_joints, names)
    def ids(group):
        return [names.index(name) - 1 for name in group]
    if names[0] == 'pelvis':
        return dict(legs=ids(LEFT_LEG + RIGHT_LEG),
                    toes=ids(['left_foot', 'right_foot']),
                    end_effectors=ids(['left_foot', 'right_foot', 'head', 'left_wrist', 'right_wrist']))
    return dict(legs=ids(LEFT_LEG + RIGHT_LEG),
                toes=ids(['LeftToeBase', 'RightToeBase']),
                end_effectors=ids(['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']))
