#!/usr/bin/env python3
"""Gripper helper for real robot and simulated mode."""

import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Gripperpick, Gripperset


class GripperClient:
    """Publish RealMan gripper commands or simulate success in Gazebo mode."""

    def __init__(self, node: Node, use_sim: bool) -> None:
        self._node = node
        self._use_sim = use_sim
        if not use_sim:
            self._pick_pub = node.create_publisher(Gripperpick, '/rm_driver/set_gripper_pick_cmd', 10)
            self._set_pub = node.create_publisher(Gripperset, '/rm_driver/set_gripper_position_cmd', 10)

    def close(self, speed: int = 300, force: int = 500) -> None:
        if self._use_sim:
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
            self._node.get_logger().info('[sim] gripper open')
            return
        msg = Gripperset()
        msg.position = position
        msg.block = True
        msg.timeout = 3
        self._set_pub.publish(msg)
