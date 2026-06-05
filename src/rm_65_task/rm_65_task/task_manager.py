#!/usr/bin/env python3
"""Gesture-driven pick and place state machine for RM65."""

import os
import yaml

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from rm_65_task.arm_controller import ArmController
from rm_65_task.gripper_client import GripperClient


class TaskManager(Node):
    """Translate gesture commands into pick-place actions."""

    STATE_IDLE = 'IDLE'
    STATE_MOVING = 'MOVING'
    STATE_WAIT_GRASP = 'WAIT_GRASP'
    STATE_WAIT_PLACE = 'WAIT_PLACE'

    def __init__(self) -> None:
        super().__init__('task_manager')
        self.declare_parameter('use_sim', True)
        self.declare_parameter('poses_file', '')
        self.declare_parameter('align_max_steps', 3)
        self.declare_parameter('align_threshold_m', 0.002)

        self._use_sim = self.get_parameter('use_sim').value
        self._align_max_steps = int(self.get_parameter('align_max_steps').value)
        self._align_threshold = float(self.get_parameter('align_threshold_m').value)
        self._state = self.STATE_IDLE
        self._active_zone = 'left'
        self._latest_offset = None
        self._busy = False

        self._poses = self._load_poses()
        self._arm = ArmController(self, self._use_sim)
        self._gripper = GripperClient(self, self._use_sim)

        self.create_subscription(String, 'gesture_cmd', self._on_gesture, 10)
        self.create_subscription(PointStamped, 'grasp_offset', self._on_grasp_offset, 10)
        mode = 'simulation' if self._use_sim else 'real robot'
        self.get_logger().info(f'Task manager ready ({mode})')

    def _load_poses(self) -> dict:
        poses_file = self.get_parameter('poses_file').value
        if not poses_file:
            raise RuntimeError('poses_file parameter is required')
        with open(poses_file, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)

    def _on_grasp_offset(self, msg: PointStamped) -> None:
        self._latest_offset = msg

    def _on_gesture(self, msg: String) -> None:
        if self._busy:
            self.get_logger().warn(f'Busy in state={self._state}, ignore command={msg.data}')
            return
        command = msg.data.strip()
        if command == 'ESTOP':
            self.get_logger().warn('Emergency stop requested by gesture')
            self._state = self.STATE_IDLE
            return
        if command == 'HOME':
            self._run_async(self._go_home)
            return
        if command == 'GOTO_LEFT':
            self._active_zone = 'left'
            self._run_async(self._goto_zone, 'left')
            return
        if command == 'GOTO_RIGHT':
            self._active_zone = 'right'
            self._run_async(self._goto_zone, 'right')
            return
        if command == 'GRASP':
            if self._state != self.STATE_WAIT_GRASP:
                self.get_logger().warn('GRASP ignored: move to a zone first')
                return
            self._run_async(self._execute_grasp)
            return
        if command == 'PLACE':
            if self._state != self.STATE_WAIT_PLACE:
                self.get_logger().warn('PLACE ignored: grasp an object first')
                return
            self._run_async(self._execute_place)

    def _run_async(self, callback, *args) -> None:
        self._busy = True
        try:
            callback(*args)
        finally:
            self._busy = False

    def _go_home(self) -> None:
        self.get_logger().info('Moving to home')
        if self._arm.move_joints(self._poses['home']):
            self._state = self.STATE_IDLE
            self._gripper.open()

    def _goto_zone(self, zone: str) -> None:
        key = f'{zone}_zone'
        self.get_logger().info(f'Moving to {key}')
        if self._arm.move_joints(self._poses[key]):
            self._state = self.STATE_WAIT_GRASP
            self._latest_offset = None

    def _execute_grasp(self) -> None:
        zone_key = f'{self._active_zone}_zone'
        pre_grasp = list(self._poses[zone_key])
        grasp = list(self._poses[zone_key])

        self._apply_alignment(pre_grasp, grasp)

        pre_grasp[1] = float(pre_grasp[1]) + float(self._poses.get('pre_grasp_lift_joint2', 0.05))
        grasp[1] = float(grasp[1]) + float(self._poses.get('grasp_lower_joint2', -0.03))

        self.get_logger().info('Opening gripper before grasp')
        self._gripper.open()
        self.get_logger().info('Moving to pre-grasp pose')
        if not self._arm.move_joints(pre_grasp):
            return
        self.get_logger().info('Descending to grasp pose')
        if not self._arm.move_joints(grasp):
            return
        self.get_logger().info('Closing gripper')
        self._gripper.close()
        self._state = self.STATE_WAIT_PLACE

    def _apply_alignment(self, pre_grasp: list, grasp: list) -> None:
        for step in range(self._align_max_steps):
            if self._latest_offset is None:
                self.get_logger().warn('No grasp offset yet, skip visual alignment')
                return
            dx = self._latest_offset.point.x
            dy = self._latest_offset.point.y
            if abs(dx) < self._align_threshold and abs(dy) < self._align_threshold:
                self.get_logger().info(f'Alignment converged in {step} steps')
                return
            pre_grasp[0] = float(pre_grasp[0]) + dx
            pre_grasp[4] = float(pre_grasp[4]) + dy
            grasp[0] = float(grasp[0]) + dx
            grasp[4] = float(grasp[4]) + dy
            self.get_logger().info(f'Alignment step {step + 1}: dx={dx:.4f}, dy={dy:.4f}')
            if not self._arm.move_joints(pre_grasp, duration_sec=2.0):
                return
            self._latest_offset = None
            rclpy.spin_once(self, timeout_sec=0.5)

    def _execute_place(self) -> None:
        place = list(self._poses['place_zone'])
        pre_place = list(place)
        pre_place[1] = float(pre_place[1]) + float(self._poses.get('pre_grasp_lift_joint2', 0.05))

        self.get_logger().info('Moving to place pre-position')
        if not self._arm.move_joints(pre_place):
            return
        self.get_logger().info('Moving to place position')
        if not self._arm.move_joints(place):
            return
        self.get_logger().info('Opening gripper to release object')
        self._gripper.open()
        self._go_home()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
