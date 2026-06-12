#!/usr/bin/env python3
"""Gripper / dexterous-hand helper for real robot and Gazebo simulation."""

from typing import Optional

from rclpy.node import Node
from rm_ros_interfaces.msg import Gripperpick, Gripperset

from rm_65_task.sim_hand_client import SimHandClient


class GripperClient:
    """Publish RealMan gripper commands, or drive the simulated dexterous hand."""

    def __init__(
        self,
        node: Node,
        use_sim: bool,
        use_dexterous_hand: Optional[bool] = None,
    ) -> None:
        self._node = node
        self._use_sim = use_sim
        if use_dexterous_hand is None:
            use_dexterous_hand = use_sim
        self._use_dexterous_hand = bool(use_dexterous_hand)
        self._sim_hand = (
            SimHandClient(node, enabled=self._use_dexterous_hand)
            if use_sim and self._use_dexterous_hand
            else None
        )
        if not use_sim:
            self._pick_pub = node.create_publisher(
                Gripperpick, '/rm_driver/set_gripper_pick_cmd', 10
            )
            self._set_pub = node.create_publisher(
                Gripperset, '/rm_driver/set_gripper_position_cmd', 10
            )

    def close(self, speed: int = 300, force: int = 500) -> None:
        if self._use_sim:
            if self._sim_hand is not None:
                self._sim_hand.close()
            else:
                self._node.get_logger().info('[sim] gripper close')
            return
        msg = Gripperpick()
        msg.speed = speed
        msg.force = force
        msg.block = True
        msg.timeout = 3
        self._pick_pub.publish(msg)

    def open(self, position: int = 900) -> None:
        if self._use_sim:
            if self._sim_hand is not None:
                self._sim_hand.open()
            else:
                self._node.get_logger().info('[sim] gripper open')
            return
        msg = Gripperset()
        msg.position = position
        msg.block = True
        msg.timeout = 3
        self._set_pub.publish(msg)
