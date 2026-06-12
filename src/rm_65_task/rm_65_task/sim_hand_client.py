#!/usr/bin/env python3
"""Drive Allegro Hand joints in Gazebo via ros2_control."""

from typing import Iterable, List

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Allegro Hand joint order (matches hand_group_controller / PAL ros2_control macro)
ALLEGRO_JOINTS = [
    'gripper_thumb_rotatory_joint',
    'gripper_thumb_flexor_1_joint',
    'gripper_thumb_flexor_2_joint',
    'gripper_thumb_flexor_3_joint',
    'gripper_finger_1_rotatory_joint',
    'gripper_finger_1_flexor_1_joint',
    'gripper_finger_1_flexor_2_joint',
    'gripper_finger_1_flexor_3_joint',
    'gripper_finger_2_rotatory_joint',
    'gripper_finger_2_flexor_1_joint',
    'gripper_finger_2_flexor_2_joint',
    'gripper_finger_2_flexor_3_joint',
    'gripper_finger_3_rotatory_joint',
    'gripper_finger_3_flexor_1_joint',
    'gripper_finger_3_flexor_2_joint',
    'gripper_finger_3_flexor_3_joint',
]

# Open / closed presets from Allegro URDF joint limits (with safety margin)
OPEN_POSE = [
    0.55, -0.10, -0.18, -0.16,
    0.0, -0.19, -0.17, -0.22,
    0.0, -0.19, -0.17, -0.22,
    0.0, -0.19, -0.17, -0.22,
]
CLOSE_POSE = [
    1.20, 1.10, 1.50, 1.55,
    0.35, 1.45, 1.55, 1.50,
    0.0, 1.45, 1.55, 1.50,
    -0.35, 1.45, 1.55, 1.50,
]


def hand_cmd_to_radians(hand_angle: Iterable[int], current: List[float]) -> List[float]:
    """Map RealMan 6-DoF hand_angle (0~1000) to Allegro 16-DoF radians."""
    values = list(hand_angle)
    if len(values) != 6:
        raise ValueError('hand_angle must contain 6 values')

    pose = list(current)
    finger_maps = [
        (13, 14, 15, values[0]),  # pinky  -> Allegro finger 3
        (9, 10, 11, values[1]),    # ring   -> Allegro finger 2
        (5, 6, 7, values[2]),       # middle -> Allegro finger 1
    ]
    for idx_a, idx_b, idx_c, raw in finger_maps:
        if raw < 0:
            continue
        ratio = max(0.0, min(1.0, int(raw) / 1000.0))
        for idx in (idx_a, idx_b, idx_c):
            pose[idx] = OPEN_POSE[idx] + ratio * (CLOSE_POSE[idx] - OPEN_POSE[idx])

    index_raw = values[3]
    if index_raw >= 0:
        ratio = max(0.0, min(1.0, int(index_raw) / 1000.0))
        pose[4] = OPEN_POSE[4] + ratio * (CLOSE_POSE[4] - OPEN_POSE[4])

    thumb_bend = values[4]
    if thumb_bend >= 0:
        ratio = max(0.0, min(1.0, int(thumb_bend) / 1000.0))
        for idx in (1, 2, 3):
            pose[idx] = OPEN_POSE[idx] + ratio * (CLOSE_POSE[idx] - OPEN_POSE[idx])

    thumb_rot = values[5]
    if thumb_rot >= 0:
        ratio = max(0.0, min(1.0, int(thumb_rot) / 1000.0))
        pose[0] = OPEN_POSE[0] + ratio * (CLOSE_POSE[0] - OPEN_POSE[0])

    return pose


class SimHandClient:
    """Publish /hand_group_controller/commands for the simulated Allegro Hand."""

    def __init__(self, node: Node, enabled: bool = True) -> None:
        self._node = node
        self._enabled = enabled
        self._current = list(OPEN_POSE)
        self._pub = node.create_publisher(
            Float64MultiArray,
            '/hand_group_controller/commands',
            10,
        )

    @property
    def current_pose(self) -> List[float]:
        return list(self._current)

    def set_pose(self, radians: List[float]) -> None:
        if not self._enabled or len(radians) != len(ALLEGRO_JOINTS):
            return
        self._current = [float(v) for v in radians]
        msg = Float64MultiArray()
        msg.data = self._current
        self._pub.publish(msg)

    def set_hand_angle(self, hand_angle: Iterable[int]) -> None:
        self.set_pose(hand_cmd_to_radians(hand_angle, self._current))

    def open(self) -> None:
        self.set_pose(OPEN_POSE)
        self._node.get_logger().info('[sim] Allegro hand open')

    def close(self) -> None:
        self.set_pose(CLOSE_POSE)
        self._node.get_logger().info('[sim] Allegro hand close')
