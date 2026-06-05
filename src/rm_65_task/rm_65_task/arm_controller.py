#!/usr/bin/env python3
"""Joint-space motion helper for simulation and real robot."""

import time
from typing import List, Optional

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ArmController:
    """Move RM65 to named joint targets."""

    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    TRAJECTORY_TOPIC = '/rm_group_controller/joint_trajectory'

    def __init__(self, node: Node, use_sim: bool, move_speed: int = 20) -> None:
        self._node = node
        self._use_sim = use_sim
        self._move_speed = move_speed
        self._current_joints: Optional[List[float]] = None
        self._publish_count = 0
        self._warned_no_feedback = False
        self._warned_no_subscriber = False
        self._logged_connected = False
        self._movej_pub = node.create_publisher(Movej, '/rm_driver/movej_cmd', 10)
        self._movej_result = None
        if not use_sim:
            node.create_subscription(Bool, '/rm_driver/movej_result', self._on_movej_result, 10)
        if use_sim:
            node.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self._action_client = ActionClient(
            node, FollowJointTrajectory, '/rm_group_controller/follow_joint_trajectory'
        )
        self._traj_pub = node.create_publisher(JointTrajectory, self.TRAJECTORY_TOPIC, 10)

    @staticmethod
    def _normalize_joint(name: str) -> str:
        return name.strip().lower()

    def _on_joint_states(self, msg: JointState) -> None:
        index = {self._normalize_joint(name): idx for idx, name in enumerate(self.JOINT_NAMES)}
        positions = [0.0] * 6
        found = 0
        for name, pos in zip(msg.name, msg.position):
            key = self._normalize_joint(name)
            if key in index:
                positions[index[key]] = float(pos)
                found += 1
        if found == 6:
            self._current_joints = positions
            self._warned_no_feedback = False

    def sim_controller_ready(self) -> bool:
        if not self._use_sim:
            return True
        return (
            self._traj_pub.get_subscription_count() > 0
            or self._action_client.server_is_ready()
        )

    def has_joint_feedback(self) -> bool:
        return self._current_joints is not None

    def _on_movej_result(self, msg: Bool) -> None:
        self._movej_result = msg.data

    def move_joints(self, joints: List[float], duration_sec: float = 4.0) -> bool:
        if len(joints) != 6:
            self._node.get_logger().error('Expected 6 joint values')
            return False
        if self._use_sim:
            self._stream_joints_sim(joints, duration_sec)
            return True
        return self._move_joints_real(joints)

    def stream_joints(self, joints: List[float], duration_sec: float = 0.2) -> None:
        """Send frequent joint updates for teleoperation without blocking."""
        if len(joints) != 6:
            return
        if self._use_sim:
            self._stream_joints_sim(joints, duration_sec)
        else:
            self._stream_joints_real(joints)

    def _warn_if_sim_not_connected(self) -> None:
        if not self._use_sim:
            return
        if not self._warned_no_subscriber:
            sub_count = self._traj_pub.get_subscription_count()
            if sub_count == 0:
                self._node.get_logger().error(
                    'Gazebo 控制器未连接：/rm_group_controller/joint_trajectory 无订阅者。'
                    '请先 killall gzserver gzclient 后重新 launch。'
                )
                self._warned_no_subscriber = True
        if not self._logged_connected and self._traj_pub.get_subscription_count() > 0:
            self._node.get_logger().info(
                f'Gazebo 控制器已连接 (轨迹订阅={self._traj_pub.get_subscription_count()})'
            )
            self._logged_connected = True
        if not self._warned_no_feedback and self._publish_count >= 20:
            if self._current_joints is None:
                self._node.get_logger().warn(
                    '未收到 /joint_states 反馈，轨迹将使用单点模式。'
                    '若机械臂仍不动，请检查 Gazebo 与 joint_state_broadcaster。'
                )
                self._warned_no_feedback = True

    def _stream_joints_sim(self, joints: List[float], duration_sec: float) -> None:
        duration_sec = max(float(duration_sec), 0.08)
        target = [float(v) for v in joints]

        msg = JointTrajectory()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.joint_names = self.JOINT_NAMES

        p1 = JointTrajectoryPoint()
        p1.positions = target
        p1.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec - int(duration_sec)) * 1e9),
        )

        if self._current_joints is not None:
            p0 = JointTrajectoryPoint()
            p0.positions = list(self._current_joints)
            p0.time_from_start = Duration(sec=0, nanosec=0)
            msg.points = [p0, p1]
        else:
            # 无反馈时只发目标点，让 JTC 从当前物理位置插值过去
            msg.points = [p1]

        self._traj_pub.publish(msg)
        self._publish_count += 1
        self._warn_if_sim_not_connected()

    @property
    def publish_count(self) -> int:
        return self._publish_count

    def _stream_joints_real(self, joints: List[float]) -> None:
        if not hasattr(self, '_canfd_pub'):
            from rm_ros_interfaces.msg import Jointpos
            self._canfd_msg_type = Jointpos
            self._canfd_pub = self._node.create_publisher(Jointpos, '/rm_driver/movej_canfd_cmd', 10)
        msg = self._canfd_msg_type()
        msg.joint = [float(v) for v in joints]
        msg.follow = True
        msg.expand = 0.0
        msg.dof = 6
        self._canfd_pub.publish(msg)

    def _move_joints_real(self, joints: List[float]) -> bool:
        self._movej_result = None
        msg = Movej()
        msg.joint = [float(v) for v in joints]
        msg.speed = self._move_speed
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self._movej_pub.publish(msg)

        deadline = time.time() + 20.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._movej_result is not None:
                return bool(self._movej_result)
        self._node.get_logger().error('Timed out waiting for movej result')
        return False
