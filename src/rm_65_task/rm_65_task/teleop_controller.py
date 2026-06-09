#!/usr/bin/env python3
"""Map tracked human arm joints to RM65 and stream motion commands."""

import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from rm_65_task.arm_controller import ArmController
from rm_65_task.gripper_client import GripperClient
from rm_65_task.teleop_mapper import TeleopMapper


class TeleopController(Node):
    """Follow human arm for reach-to-grasp teleoperation."""

    REQUIRED_HUMAN_KEYS = [
        'upper_arm_elev',
        'upper_arm_yaw',
        'elbow_bend',
        'wrist_pitch',
        'palm_open',
    ]

    def __init__(self) -> None:
        super().__init__('teleop_controller')
        self.declare_parameter('use_sim', True)
        self.declare_parameter('mapping_file', '')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('enabled', True)

        self._use_sim = self.get_parameter('use_sim').value
        self._enabled = self.get_parameter('enabled').value
        self._config = self._load_mapping()
        self._merge_calibration()
        teleop_mode = str(self._config.get('teleop_mode', 'joints'))
        if teleop_mode != 'joints':
            self.get_logger().warn(
                f'teleop_mode={teleop_mode} not implemented; using joints mapping'
            )
        self._mapper = TeleopMapper(self._config)
        self._latest_human = None
        self._gripper_closed = False
        self._teleop_active = False
        self._follow_enabled = False
        self._override_joints = self._mapper.go_home()
        self._stream_count = 0
        self._diag_count = 0
        self._follow_started = False

        self._arm = ArmController(self, self._use_sim, move_speed=15)
        self._gripper = GripperClient(self, self._use_sim)
        self._robot_joints_pub = self.create_publisher(JointState, 'robot_arm_joints', 10)

        self.create_subscription(JointState, 'human_arm_joints', self._on_human_joints, 10)
        self.create_subscription(String, 'gesture_cmd', self._on_gesture_cmd, 10)
        self.create_subscription(String, 'teleop_status', self._on_teleop_status, 10)

        rate = float(self._config.get('stream_rate_hz', 20.0))
        self._timer = self.create_timer(1.0 / rate, self._stream_to_robot)
        self._diag_timer = self.create_timer(3.0, self._diagnose)
        self._ready_duration = float(self._config.get('ready_pose_duration_sec', 0.8))
        self._follow_duration = float(self._config.get('trajectory_duration_sec', 0.25))
        mode = 'simulation' if self._use_sim else 'real robot'
        self.get_logger().info(
            f'Teleop controller ready ({mode}). '
            'Robot holds VERTICAL ready pose. '
            'Raise YOUR arm straight up to activate follow.'
        )

    def _load_mapping(self) -> dict:
        mapping_file = self.get_parameter('mapping_file').value
        if not mapping_file:
            raise RuntimeError('mapping_file parameter is required')
        with open(mapping_file, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)

    def _merge_calibration(self) -> None:
        cal_file = self.get_parameter('calibration_file').value
        if not cal_file:
            from ament_index_python.packages import get_package_share_directory
            import os
            try:
                share = get_package_share_directory('rm_65_task')
                cal_file = os.path.join(share, 'config', 'arm_teleop_calibration.yaml')
            except Exception:
                return
        try:
            with open(cal_file, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            return
        cal = data.get('calibration')
        if cal:
            self._config['calibration'] = cal
            self.get_logger().info(f'Loaded teleop calibration: {cal_file}')

    def _set_active(self, active: bool, reason: str) -> None:
        if active == self._teleop_active:
            return
        self._teleop_active = active
        self._follow_enabled = active
        if active:
            self._mapper.go_home()
            self.get_logger().info(f'Follow ACTIVE ({reason}) -> robot vertical sync')
        else:
            self._mapper.set_grasp_mode(False)
            self._override_joints = self._mapper.go_home()
            self._latest_human = None
            self._follow_started = False
            self._gripper.open()
            self._gripper_closed = False
            self.get_logger().info(f'Follow WAITING ({reason})')

    def _on_human_joints(self, msg: JointState) -> None:
        frame = (msg.header.frame_id or '').strip().lower()
        # 仅用 active 帧激活；不要用 waiting 帧取消（避免消息队列乱序把跟随关掉）
        if frame == 'active':
            self._set_active(True, 'human_arm_joints')

        values = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        if not all(key in values for key in self.REQUIRED_HUMAN_KEYS):
            return

        if not self._teleop_active:
            return

        self._latest_human = values

    def _on_teleop_status(self, msg: String) -> None:
        status = msg.data.strip()
        if status == 'ACTIVE':
            self._set_active(True, 'teleop_status')
        elif status == 'WAITING':
            self._set_active(False, 'teleop_status')

    def _on_gesture_cmd(self, msg: String) -> None:
        cmd = msg.data.strip()
        if cmd == 'RELEASE':
            self._mapper.set_grasp_mode(False)
            self._gripper.open()
            self._gripper_closed = False
            self.get_logger().info('Gesture RELEASE -> gripper open (keep following)')
        elif cmd == 'HOME':
            self._mapper.set_grasp_mode(False)
            self._override_joints = self._mapper.go_home()
            self._gripper.open()
            self._gripper_closed = False
            self.get_logger().info('Gesture HOME -> robot_home')
        elif cmd == 'GRASP':
            self._mapper.set_grasp_mode(True)
            if not self._gripper_closed:
                self._gripper.close()
                self._gripper_closed = True
            self.get_logger().info('Gesture GRASP -> approach mode + gripper close')
        elif cmd == 'PLACE':
            self._mapper.set_grasp_mode(False)
            self._override_joints = self._mapper.retract_from_grasp()
            self._gripper.open()
            self._gripper_closed = False
            self.get_logger().info('Gesture PLACE -> retract + open gripper')
        elif cmd == 'ESTOP':
            self._follow_enabled = False
            self.get_logger().warn('Gesture ESTOP -> follow paused')

    def _update_gripper(self, palm_open: float) -> None:
        close_th = float(self._config.get('gripper_close_threshold', 0.30))
        open_th = float(self._config.get('gripper_open_threshold', 0.60))

        if palm_open < close_th:
            self._mapper.set_grasp_mode(True)
            if not self._gripper_closed:
                self._gripper.close()
                self._gripper_closed = True
                self.get_logger().info('Fist -> approach + gripper close')
        elif palm_open > open_th:
            self._mapper.set_grasp_mode(False)
            if self._gripper_closed:
                self._gripper.open()
                self._gripper_closed = False
                self.get_logger().info('Open palm -> gripper open')

    def _diagnose(self) -> None:
        self._diag_count += 1
        if self._diag_count == 1 or self._diag_count % 5 == 0:
            human = '有' if self._latest_human is not None else '无'
            ctrl = '就绪' if self._arm.sim_controller_ready() else '未就绪'
            fb = '有' if self._arm.has_joint_feedback() else '无'
            active = '已激活' if self._teleop_active else '未激活(请抬臂或按C)'
            subs = self._arm._traj_pub.get_subscription_count()
            self.get_logger().info(
                f'[{active}] 人体数据={human} 控制器={ctrl} 关节反馈={fb} '
                f'轨迹订阅={subs} 已发轨迹={self._arm.publish_count}次'
            )
            if subs == 0 and self._use_sim:
                self.get_logger().error(
                    'Gazebo 未连接：请用 sim_arm_teleop_follow.launch.py 启动，'
                    '并确认 Gazebo 里机械臂在动（不是只看 RViz）'
                )
            if not self._teleop_active:
                self.get_logger().warn('机械臂竖直待命：请竖臂伸直朝上，达标后自动激活')
            elif self._latest_human is None:
                self.get_logger().warn('已激活但无人体数据：检查摄像头与追踪臂(1/2)')

    def _log_follow_debug(self, robot_joints) -> None:
        self._stream_count += 1
        if not self._follow_started:
            self._follow_started = True
            self.get_logger().info(
                '开始跟随 -> robot [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
                % tuple(robot_joints)
            )
        if self._stream_count % 20 != 0 or self._latest_human is None:
            return
        h = self._latest_human
        self.get_logger().info(
            'Following | human elev=%.2f yaw=%.2f elbow=%.2f wrist_p=%.2f -> robot '
            '[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]'
            % (
                h['upper_arm_elev'], h['upper_arm_yaw'], h['elbow_bend'],
                h['wrist_pitch'],
                *robot_joints,
            )
        )

    def _publish_robot_joints(self, joints) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint{i}' for i in range(1, 7)]
        msg.position = [float(v) for v in joints]
        self._robot_joints_pub.publish(msg)

    def _stream_to_robot(self) -> None:
        if not self._enabled:
            return

        if self._override_joints is not None:
            self._arm.stream_joints(self._override_joints, duration_sec=2.0)
            self._publish_robot_joints(self._override_joints)
            self._override_joints = None
            return

        if not self._teleop_active:
            ready = self._mapper.go_home()
            self._arm.stream_joints(ready, duration_sec=self._ready_duration)
            self._publish_robot_joints(ready)
            return

        if not self._follow_enabled:
            return

        if self._latest_human is None:
            return

        robot_joints = self._mapper.map_joints(self._latest_human)
        self._arm.stream_joints(robot_joints, duration_sec=self._follow_duration)
        self._publish_robot_joints(robot_joints)
        self._log_follow_debug(robot_joints)
        # 夹爪仅由 GRASP / RELEASE 手势控制，避免 palm_open 抖动反复触发


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
