"""Anatomical joint groups independent of the serialized skeleton order."""
from bvh_export import DEFAULT_JOINT_NAMES

GAIT_JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
]
LEFT_LEG = ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
RIGHT_LEG = ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']


def joint_names(n_joints, names=None):
    if names is None:
        if n_joints == len(GAIT_JOINT_NAMES):
            names = GAIT_JOINT_NAMES
        elif n_joints == len(DEFAULT_JOINT_NAMES):
            names = DEFAULT_JOINT_NAMES
        else:
            raise ValueError('Unknown skeleton: supply skeleton_joint_names explicitly')
    names = list(names)
    if len(names) != n_joints or len(set(names)) != n_joints or names[0] != 'Hips':
        raise ValueError('Skeleton names must be unique, match the pose width, and start with Hips')
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
    return dict(legs=ids(LEFT_LEG + RIGHT_LEG),
                toes=ids(['LeftToeBase', 'RightToeBase']),
                end_effectors=ids(['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']))
