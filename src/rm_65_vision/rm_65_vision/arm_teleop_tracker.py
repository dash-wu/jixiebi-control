#!/usr/bin/env python3
"""Track human upper arm, forearm and palm from PC camera using MediaPipe Tasks API."""

import math
import os
from typing import Dict, Optional, Tuple

import cv2
import yaml
import mediapipe as mp
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from rm_65_vision.gesture_classifier import GestureClassifier, GestureDebouncer
from rm_65_vision.ui_text import render_panel
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class ArmTeleopTracker(Node):
    """Estimate human arm joint angles and publish them for robot teleoperation."""

    RIGHT_ARM = {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24}
    LEFT_ARM = {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23}
    PANEL_WIDTH = 320

    COLOR_UPPER = (255, 160, 40)   # BGR 大臂
    COLOR_FORE = (40, 160, 255)    # BGR 小臂
    COLOR_PALM = (40, 255, 160)    # BGR 手掌

    JOINT_NAMES = [
        'upper_arm_yaw',
        'upper_arm_elev',
        'elbow_bend',
        'wrist_pitch',
        'wrist_roll',
        'palm_open',
    ]

    JOINT_LABELS = {
        'upper_arm_yaw': '大臂左右',
        'upper_arm_elev': '大臂抬升',
        'elbow_bend': '肘部弯曲',
        'wrist_pitch': '手腕俯仰',
        'wrist_roll': '手掌旋转',
        'palm_open': '手掌张开',
    }

    ROBOT_JOINT_NAMES = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    ]
    ROBOT_JOINT_LABELS = {
        'joint1': 'J1 基座',
        'joint2': 'J2 大臂',
        'joint3': 'J3 肘部',
        'joint4': 'J4 腕旋',
        'joint5': 'J5 腕俯',
        'joint6': 'J6 末端',
    }

    def __init__(self) -> None:
        super().__init__('arm_teleop_tracker')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('tracking_side', 'right')
        self.declare_parameter('smoothing', 0.35)
        self.declare_parameter('show_debug', True)
        self.declare_parameter('publish_ros', True)
        self.declare_parameter('min_visibility', 0.5)
        self.declare_parameter('model_path', '')
        self.declare_parameter('gesture_debounce_frames', 6)
        self.declare_parameter('config_file', '')

        self._camera_id = self.get_parameter('camera_id').value
        self._smoothing = float(self.get_parameter('smoothing').value)
        self._show_debug = self.get_parameter('show_debug').value
        self._publish_ros = self.get_parameter('publish_ros').value
        self._min_visibility = float(self.get_parameter('min_visibility').value)

        side = self.get_parameter('tracking_side').value.lower()
        self._tracking_side = side
        self._arm_ids = self.RIGHT_ARM if side == 'right' else self.LEFT_ARM

        self._publisher = self.create_publisher(JointState, 'human_arm_joints', 10)
        self._gesture_pub = self.create_publisher(String, 'hand_gesture', 10)
        self._gesture_cmd_pub = self.create_publisher(String, 'gesture_cmd', 10)
        self._status_pub = self.create_publisher(String, 'teleop_status', 10)
        self._activation_cfg = self._load_activation_config()
        self._active_robot_indices = self._load_active_robot_indices()
        self._robot_actual: Optional[Dict[str, float]] = None
        self._robot_target: Optional[Dict[str, float]] = None
        self._teleop_active = False
        self._activation_hold = 0
        self._status_published = False
        self._gesture_debouncer = GestureDebouncer(
            int(self.get_parameter('gesture_debounce_frames').value)
        )
        self._current_gesture = 'None'
        self._current_gesture_raw = 'None'
        self._reference: Optional[Tuple[float, ...]] = None
        self._filtered: Optional[Tuple[float, ...]] = None
        self._last_raw: Optional[Dict[str, float]] = None
        self._frames_detected = 0
        self._frames_total = 0
        self._timestamp_ms = 0

        model_path = self.get_parameter('model_path').value
        if not model_path:
            pkg_share = get_package_share_directory('rm_65_vision')
            model_path = os.path.join(pkg_share, 'models', 'holistic_landmarker.task')
        if not os.path.isfile(model_path):
            raise RuntimeError(f'MediaPipe model not found: {model_path}')

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_landmarks_confidence=0.6,
            min_hand_landmarks_confidence=0.6,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)

        self._cap = self._open_camera(self._camera_id)
        if self._cap is None:
            raise RuntimeError(
                f'Cannot open PC camera id={self._camera_id}. '
                'Try: ros2 launch rm_65_vision arm_teleop_debug.launch.py camera_id:=1'
            )

        self._timer = self.create_timer(0.033, self._process_frame)
        self._log_timer = self.create_timer(1.0, self._log_status)
        self.create_subscription(JointState, 'joint_states', self._on_robot_joint_states, 10)
        self.create_subscription(JointState, 'robot_arm_joints', self._on_robot_arm_command, 10)
        self.get_logger().info(
            f'Arm tracker ready (MediaPipe Tasks). Model: {model_path}\n'
            'Raise arm straight up to activate follow. Keys: [R]=reset [1]=right [2]=left [Q]=quit'
        )

    def _load_activation_config(self) -> dict:
        config_file = self.get_parameter('config_file').value
        defaults = {
            'min_upper_arm_elev_deg': 140.0,
            'max_elbow_bend_deg': 28.0,
            'min_extension_ratio': 1.55,
            'min_elbow_raise': 0.015,
            'min_wrist_raise': 0.015,
            'min_extension_override_raise': 1.75,
            'hold_frames': 12,
        }
        if not config_file:
            return defaults
        with open(config_file, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        activation = data.get('activation', {})
        merged = {**defaults, **activation}
        if 'min_forearm_flex_deg' in activation and 'min_upper_arm_elev_deg' not in activation:
            merged['min_upper_arm_elev_deg'] = float(activation['min_forearm_flex_deg']) - 20.0
        return merged

    def _load_active_robot_indices(self) -> set:
        config_file = self.get_parameter('config_file').value
        if not config_file:
            return set(range(6))
        with open(config_file, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        raw = data.get('active_joint_indices')
        if raw is None:
            return set(range(6))
        return {int(i) for i in raw}

    @staticmethod
    def _parse_robot_joint_state(msg: JointState) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, pos in zip(msg.name, msg.position):
            key = name.strip().lower()
            if key in {n.lower() for n in ArmTeleopTracker.ROBOT_JOINT_NAMES}:
                values[key] = float(pos)
        return values

    def _on_robot_joint_states(self, msg: JointState) -> None:
        parsed = self._parse_robot_joint_state(msg)
        if len(parsed) == 6:
            self._robot_actual = parsed

    def _on_robot_arm_command(self, msg: JointState) -> None:
        parsed = self._parse_robot_joint_state(msg)
        if len(parsed) == 6:
            self._robot_target = parsed

    def _publish_teleop_status(self, status: str) -> None:
        if not self._publish_ros:
            return
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _set_teleop_active(self, active: bool) -> None:
        if active and not self._teleop_active:
            self._teleop_active = True
            self._publish_teleop_status('ACTIVE')
            self.get_logger().info('Activation pose reached -> teleop ACTIVE (vertical zero set)')
        elif not active and self._teleop_active:
            self._teleop_active = False
            self._publish_teleop_status('WAITING')
            self.get_logger().info('Teleop reset -> WAITING for activation pose')

    def _activation_metrics(
        self,
        raw: Dict[str, float],
        shoulder,
        elbow,
        wrist,
    ) -> Dict[str, float]:
        upper_len = math.hypot(elbow.x - shoulder.x, elbow.y - shoulder.y)
        arm_span = math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y)
        return {
            'elbow_bend_deg': math.degrees(raw['elbow_bend']),
            'upper_arm_elev_deg': math.degrees(raw['upper_arm_elev']),
            'extension_ratio': arm_span / max(upper_len, 1e-6),
            'elbow_raise': float(shoulder.y - elbow.y),
            'wrist_raise': float(elbow.y - wrist.y),
        }

    def _is_activation_pose(self, metrics: Dict[str, float]) -> bool:
        """Require human arm straight up before teleop starts."""
        cfg = self._activation_cfg
        elbow_ok = metrics['elbow_bend_deg'] <= float(cfg['max_elbow_bend_deg'])
        elev_ok = metrics['upper_arm_elev_deg'] >= float(cfg['min_upper_arm_elev_deg'])
        ext_ok = metrics['extension_ratio'] >= float(cfg['min_extension_ratio'])
        vert_ok = (
            metrics['elbow_raise'] >= float(cfg['min_elbow_raise'])
            and metrics['wrist_raise'] >= float(cfg['min_wrist_raise'])
        )
        ext_override = float(cfg.get('min_extension_override_raise', 1.75))
        strong_reach = (
            metrics['extension_ratio'] >= ext_override
            and elbow_ok
            and vert_ok
        )
        return elbow_ok and ext_ok and vert_ok and (elev_ok or strong_reach)

    def _open_camera(self, preferred_id: int):
        width = self.get_parameter('frame_width').value
        height = self.get_parameter('frame_height').value
        candidates = [preferred_id] + [i for i in range(4) if i != preferred_id]
        for cam_id in candidates:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                self.get_logger().info(f'Using camera id={cam_id}')
                return cap
            cap.release()
        return None

    @staticmethod
    def _lm_to_vec(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=float)

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.acos(cos_val))

    @staticmethod
    def _upper_arm_elevation_rad(upper_arm: np.ndarray) -> float:
        """Elevation in image plane: pi=straight up, pi/2=horizontal.

        Ignore depth (z) so raising the arm beside the body is not penalized
        when the user faces the camera at an angle.
        """
        sideward = abs(float(upper_arm[0]))
        upward = max(-float(upper_arm[1]), 0.0)
        if sideward < 1e-8 and upward < 1e-8:
            return math.pi / 2
        return math.atan2(upward, sideward) + math.pi / 2

    def _visible(self, pose_landmarks, name: str) -> bool:
        idx = self._arm_ids[name]
        lm = pose_landmarks[idx]
        return getattr(lm, 'visibility', 1.0) >= self._min_visibility

    def _get_hand_landmarks(self, result):
        """Return 21-point hand landmarks for the tracked arm side, if valid."""
        hand = result.right_hand_landmarks if self._tracking_side == 'right' else result.left_hand_landmarks
        if GestureClassifier.is_valid(hand):
            return hand
        # 追踪侧未检测到时，尝试另一只手（侧身时可能只检测到一只）
        fallback = result.left_hand_landmarks if self._tracking_side == 'right' else result.right_hand_landmarks
        if GestureClassifier.is_valid(fallback):
            return fallback
        return None

    def _compute_arm_angles(
        self,
        pose_landmarks,
        hand_landmarks,
        palm_open: float,
    ) -> Dict[str, float]:
        shoulder = pose_landmarks[self._arm_ids['shoulder']]
        elbow = pose_landmarks[self._arm_ids['elbow']]
        wrist = pose_landmarks[self._arm_ids['wrist']]

        upper_arm = self._lm_to_vec(elbow) - self._lm_to_vec(shoulder)
        forearm = self._lm_to_vec(wrist) - self._lm_to_vec(elbow)

        upper_arm_yaw = math.atan2(upper_arm[0], upper_arm[2] + 1e-8)
        upper_arm_elev = self._upper_arm_elevation_rad(upper_arm)
        elbow_bend = self._angle_between(upper_arm, forearm)

        hand_dir = forearm.copy()
        palm_span = np.array([1.0, 0.0, 0.0])
        if GestureClassifier.is_valid(hand_landmarks):
            wrist_h = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            index_mcp = hand_landmarks[5]
            pinky_mcp = hand_landmarks[17]
            hand_dir = self._lm_to_vec(middle_tip) - self._lm_to_vec(wrist_h)
            palm_span = self._lm_to_vec(index_mcp) - self._lm_to_vec(pinky_mcp)

        wrist_pitch = self._angle_between(forearm, hand_dir)
        wrist_roll = math.atan2(palm_span[0], -palm_span[1] + 1e-8)

        return {
            'upper_arm_yaw': upper_arm_yaw,
            'upper_arm_elev': upper_arm_elev,
            'elbow_bend': elbow_bend,
            'wrist_pitch': wrist_pitch,
            'wrist_roll': wrist_roll,
            'palm_open': palm_open,
        }

    @staticmethod
    def _palm_openness(hand_landmarks, is_right_arm: bool) -> float:
        if not GestureClassifier.is_valid(hand_landmarks):
            return 0.5
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = 0
        for tip, pip in zip(tips, pips):
            if hand_landmarks[tip].y < hand_landmarks[pip].y:
                extended += 1
        if is_right_arm:
            thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x
        else:
            thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x
        if thumb_extended:
            extended += 1
        return extended / 5.0

    def _dict_to_tuple(self, values: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(values[name] for name in self.JOINT_NAMES)

    def _tuple_to_dict(self, values: Tuple[float, ...]) -> Dict[str, float]:
        return {name: values[i] for i, name in enumerate(self.JOINT_NAMES)}

    def _apply_reference(self, values: Dict[str, float]) -> Dict[str, float]:
        if self._reference is None:
            return values
        ref = self._tuple_to_dict(self._reference)
        angular = {'upper_arm_yaw', 'wrist_pitch', 'wrist_roll'}
        result = {}
        for key in self.JOINT_NAMES:
            delta = values[key] - ref[key]
            if key in angular:
                delta = math.atan2(math.sin(delta), math.cos(delta))
            result[key] = delta
        return result

    def _smooth(self, values: Dict[str, float]) -> Dict[str, float]:
        tup = self._dict_to_tuple(values)
        if self._filtered is None:
            self._filtered = tup
            return values
        alpha = self._smoothing
        smoothed = tuple(alpha * v + (1.0 - alpha) * f for v, f in zip(tup, self._filtered))
        self._filtered = smoothed
        return self._tuple_to_dict(smoothed)

    def _publish(self, values: Dict[str, float]) -> None:
        if not self._publish_ros:
            return
        if not self._teleop_active:
            values = {name: 0.0 for name in self.JOINT_NAMES}
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'active' if self._teleop_active else 'waiting'
        msg.name = self.JOINT_NAMES
        msg.position = [values[n] for n in self.JOINT_NAMES]
        self._publisher.publish(msg)

    def _publish_gesture(self, gesture: str, changed: bool) -> None:
        if not self._publish_ros:
            return
        name_msg = String()
        name_msg.data = gesture
        self._gesture_pub.publish(name_msg)

        if changed and gesture != 'None':
            cmd = GestureClassifier.to_command(gesture)
            if cmd:
                cmd_msg = String()
                cmd_msg.data = cmd
                self._gesture_cmd_pub.publish(cmd_msg)
                self.get_logger().info(
                    f'Gesture: {GestureClassifier.label_cn(gesture)} -> {cmd}'
                )

    @staticmethod
    def _to_pixel(frame, x: float, y: float) -> Tuple[int, int]:
        h, w = frame.shape[:2]
        return int(x * w), int(y * h)

    def _draw_arm_only(self, video, shoulder, elbow, wrist, hand_landmarks) -> None:
        """Draw only the tracked arm: 3 segments + 4 joint dots, no extra skeleton."""
        pts = {
            'shoulder': self._to_pixel(video, shoulder.x, shoulder.y),
            'elbow': self._to_pixel(video, elbow.x, elbow.y),
            'wrist': self._to_pixel(video, wrist.x, wrist.y),
        }
        if GestureClassifier.is_valid(hand_landmarks):
            mid = hand_landmarks[12]
            pts['palm'] = self._to_pixel(video, mid.x, mid.y)
        else:
            pts['palm'] = pts['wrist']

        cv2.line(video, pts['shoulder'], pts['elbow'], self.COLOR_UPPER, 6, cv2.LINE_AA)
        cv2.line(video, pts['elbow'], pts['wrist'], self.COLOR_FORE, 6, cv2.LINE_AA)
        cv2.line(video, pts['wrist'], pts['palm'], self.COLOR_PALM, 5, cv2.LINE_AA)

        for pt in pts.values():
            cv2.circle(video, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(video, pt, 5, (40, 40, 40), -1, cv2.LINE_AA)

    def _draw_hand_fingers(self, video, hand_landmarks, gesture: str) -> None:
        """Draw fingertip dots and highlight active gesture fingers."""
        if not GestureClassifier.is_valid(hand_landmarks):
            return
        tip_ids = [4, 8, 12, 16, 20]
        wrist = self._to_pixel(video, hand_landmarks[0].x, hand_landmarks[0].y)
        highlight = (0, 220, 255) if gesture not in ('None', 'Closed_Fist') else (180, 180, 180)
        for tip_id in tip_ids:
            tip = hand_landmarks[tip_id]
            px = self._to_pixel(video, tip.x, tip.y)
            cv2.circle(video, px, 5, highlight, -1, cv2.LINE_AA)
            cv2.line(video, wrist, px, (60, 60, 60), 1, cv2.LINE_AA)

    def _render_ui(
        self,
        frame,
        values: Dict[str, float],
        status: str,
        hand_ok: bool,
        shoulder,
        elbow,
        wrist,
        hand_landmarks,
        gesture: str,
        gesture_raw: str,
        activation_progress: float,
        activation_metrics: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Video on the left, Chinese-capable info panel on the right."""
        h, w = frame.shape[:2]
        video = frame.copy()
        self._draw_arm_only(video, shoulder, elbow, wrist, hand_landmarks)
        self._draw_hand_fingers(video, hand_landmarks, gesture)

        canvas_h = max(h, 760)
        canvas = np.full((canvas_h, w + self.PANEL_WIDTH, 3), 28, dtype=np.uint8)
        canvas[:h, :w] = video

        side_text = '右臂' if self._tracking_side == 'right' else '左臂'
        status_cn = '追踪中' if status == 'tracking' else '可见度低'
        hand_cn = '已检测' if hand_ok else '未检测'
        gesture_cn = GestureClassifier.label_cn(gesture)
        cmd = GestureClassifier.to_command(gesture)
        if self._teleop_active:
            follow_cn = '跟随已激活'
            follow_color = (0, 220, 120)
        else:
            follow_cn = '等待：请竖臂伸直朝上'
            follow_color = (0, 180, 255)

        lines = [
            ('手臂 + 手势调试', 8, 22, (220, 220, 220)),
            (f'追踪: {side_text}', 44, 17, (0, 220, 255)),
            (follow_cn, 68, 17, follow_color),
            (f'状态: {status_cn}', 92, 17, (0, 220, 120) if status == 'tracking' else (0, 140, 255)),
            (f'手掌: {hand_cn}', 116, 17, (0, 220, 120) if hand_ok else (80, 80, 255)),
        ]
        if not self._teleop_active:
            pct = int(activation_progress * 100)
            lines.append((f'激活进度: {pct}%', 140, 16, (255, 220, 80)))
            if activation_metrics is not None:
                cfg = self._activation_cfg
                flex_ok = activation_metrics['elbow_bend_deg'] <= float(cfg['max_elbow_bend_deg'])
                elev_ok = activation_metrics['upper_arm_elev_deg'] >= float(cfg['min_upper_arm_elev_deg'])
                ext_ok = activation_metrics['extension_ratio'] >= float(cfg['min_extension_ratio'])
                vert_ok = (
                    activation_metrics['elbow_raise'] >= float(cfg['min_elbow_raise'])
                    and activation_metrics['wrist_raise'] >= float(cfg['min_wrist_raise'])
                )
                lines.append((
                    f"肘直: {activation_metrics['elbow_bend_deg']:.0f}° "
                    f"({'OK' if flex_ok else '需更直'})",
                    162, 14, (0, 220, 120) if flex_ok else (80, 80, 255),
                ))
                lines.append((
                    f"竖臂: {activation_metrics['upper_arm_elev_deg']:.0f}° "
                    f"({'OK' if elev_ok else '需竖直'})",
                    182, 14, (0, 220, 120) if elev_ok else (80, 80, 255),
                ))
                lines.append((
                    f"向上: 肘{activation_metrics['elbow_raise']:.2f} "
                    f"腕{activation_metrics['wrist_raise']:.2f} "
                    f"({'OK' if vert_ok else '需朝上'})",
                    202, 14, (0, 220, 120) if vert_ok else (80, 80, 255),
                ))
                lines.append((
                    f"伸展: {activation_metrics['extension_ratio']:.2f} "
                    f"({'OK' if ext_ok else '不足'})",
                    222, 14, (0, 220, 120) if ext_ok else (80, 80, 255),
                ))
                y_start = 248
            else:
                y_start = 168
        else:
            y_start = 140

        lines.extend([
            ('手势识别', y_start, 17, (180, 180, 180)),
            (gesture_cn, y_start + 28, 28, (0, 255, 255)),
            (f'({gesture})', y_start + 62, 14, (160, 160, 160)),
        ])
        y = y_start + 82
        if gesture_raw != gesture:
            lines.append((f'原始: {gesture_raw}', y, 14, (120, 120, 120)))
            y += 22
        if cmd:
            lines.append((f'指令: {cmd}', y, 16, (0, 200, 120)))
            y += 28

        lines.append(('图例', y, 17, (180, 180, 180)))
        legend_y_start = y + 26
        legend = [('大臂', self.COLOR_UPPER), ('小臂', self.COLOR_FORE), ('手掌', self.COLOR_PALM)]
        for i, (label, _color) in enumerate(legend):
            lines.append((label, legend_y_start + i * 24 - 14, 16, (200, 200, 200)))

        y = legend_y_start + len(legend) * 24 + 12
        lines.append(('人体关节', y, 17, (180, 180, 180)))
        y += 28
        for name in self.JOINT_NAMES:
            val = values[name]
            if name == 'palm_open':
                text = f'{self.JOINT_LABELS[name]}: {val:.2f}'
            else:
                text = f'{self.JOINT_LABELS[name]}: {math.degrees(val):+.1f}°'
            lines.append((text, y, 16, (235, 235, 235)))
            y += 22

        y += 6
        lines.append(('机械臂关节 (度)', y, 17, (180, 180, 180)))
        y += 24
        lines.append(('名称    实际      目标', y, 14, (140, 140, 140)))
        y += 20
        for idx, name in enumerate(self.ROBOT_JOINT_NAMES):
            key = name.lower()
            actual = self._robot_actual.get(key) if self._robot_actual else None
            target = self._robot_target.get(key) if self._robot_target else None
            if actual is None and target is None:
                text = f'{self.ROBOT_JOINT_LABELS[name]}  --      --'
                color = (100, 100, 100)
            else:
                act_s = f'{math.degrees(actual):+6.1f}' if actual is not None else '   --  '
                tgt_s = f'{math.degrees(target):+6.1f}' if target is not None else '   --  '
                text = f'{self.ROBOT_JOINT_LABELS[name]} {act_s} {tgt_s}'
                if idx in self._active_robot_indices:
                    color = (120, 255, 180)
                else:
                    color = (160, 160, 160)
            lines.append((text, y, 14, color))
            y += 20

        y += 6
        lines.append(('快捷键:', y, 16, (140, 140, 140)))
        y += 22
        for hint in ['竖臂朝上自动激活', 'R  重置', '1/2  切换左右臂', 'Q  退出']:
            lines.append((hint, y, 15, (120, 120, 120)))
            y += 20

        panel = render_panel(self.PANEL_WIDTH, canvas_h, lines)
        canvas[:canvas_h, w:w + self.PANEL_WIDTH] = panel
        for i, (_label, color) in enumerate(legend):
            ly = legend_y_start + i * 24
            cv2.line(canvas, (w + 16, ly - 4), (w + 44, ly - 4), color, 4, cv2.LINE_AA)
        return canvas

    def _update_gesture(self, hand_landmarks) -> None:
        is_right = self._tracking_side == 'right'
        raw = GestureClassifier.classify(hand_landmarks, is_right)
        self._current_gesture_raw = raw
        stable, changed = self._gesture_debouncer.update(raw)
        self._current_gesture = stable
        self._publish_gesture(stable, changed)

    def _log_status(self) -> None:
        if self._frames_total == 0:
            return
        rate = 100.0 * self._frames_detected / self._frames_total
        if self._last_raw is None:
            self.get_logger().info(f'Detection rate: {rate:.0f}% (move into frame, side view)')
            return
        parts = [f'{k}={math.degrees(self._last_raw[k]):.1f}' for k in self.JOINT_NAMES[:5]]
        parts.append(f'palm={self._last_raw["palm_open"]:.2f}')
        parts.append(f'gesture={GestureClassifier.label_cn(self._current_gesture)}')
        self.get_logger().info(f'Detection {rate:.0f}% | ' + ' '.join(parts))

    def _switch_side(self, side: str) -> None:
        self._tracking_side = side
        self._arm_ids = self.RIGHT_ARM if side == 'right' else self.LEFT_ARM
        self._reference = None
        self._filtered = None
        self._activation_hold = 0
        self._teleop_active = False
        self._publish_teleop_status('WAITING')
        self._gesture_debouncer = GestureDebouncer(
            int(self.get_parameter('gesture_debounce_frames').value)
        )
        self._current_gesture = 'None'
        self.get_logger().info(f'Switched to {side} arm tracking')

    def _process_frame(self) -> None:
        ok, frame = self._cap.read()
        self._frames_total += 1
        if not ok:
            self.get_logger().warn('Failed to read PC camera frame')
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._timestamp_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            if self._show_debug:
                blank = np.full((480, 640 + self.PANEL_WIDTH, 3), 28, dtype=np.uint8)
                panel = render_panel(self.PANEL_WIDTH, 480, [
                    ('未检测到人体', 40, 22, (80, 80, 255)),
                    ('请侧身对准摄像头', 80, 17, (180, 180, 180)),
                    ('确保肩-肘-腕-手在画面内', 108, 17, (180, 180, 180)),
                ])
                blank[:, 640:] = panel
                cv2.imshow('arm_teleop_tracker', blank)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    rclpy.shutdown()
            return

        pose_landmarks = result.pose_landmarks
        visible = all(self._visible(pose_landmarks, n) for n in ('shoulder', 'elbow', 'wrist'))
        status = 'tracking' if visible else 'low visibility'
        if visible:
            self._frames_detected += 1

        shoulder = pose_landmarks[self._arm_ids['shoulder']]
        elbow = pose_landmarks[self._arm_ids['elbow']]
        wrist = pose_landmarks[self._arm_ids['wrist']]

        hand_landmarks = self._get_hand_landmarks(result)
        palm_open = 0.5
        if hand_landmarks:
            palm_open = self._palm_openness(hand_landmarks, self._tracking_side == 'right')

        self._update_gesture(hand_landmarks)

        raw = self._compute_arm_angles(pose_landmarks, hand_landmarks, palm_open)
        self._last_raw = raw.copy()

        activation_progress = 0.0
        activation_metrics = None
        if not self._status_published:
            self._publish_teleop_status('WAITING')
            self._status_published = True

        if visible:
            activation_metrics = self._activation_metrics(raw, shoulder, elbow, wrist)

        if not self._teleop_active and visible and activation_metrics is not None:
            if self._is_activation_pose(activation_metrics):
                self._activation_hold += 1
                hold_frames = int(self._activation_cfg['hold_frames'])
                activation_progress = min(1.0, self._activation_hold / hold_frames)
                if self._activation_hold >= hold_frames:
                    self._reference = self._dict_to_tuple(raw)
                    self._filtered = None
                    self._set_teleop_active(True)
            else:
                self._activation_hold = 0

        values = self._smooth(self._apply_reference(raw))
        self._publish(values)

        if self._show_debug:
            ui = self._render_ui(
                frame, values, status, hand_landmarks is not None,
                shoulder, elbow, wrist, hand_landmarks,
                self._current_gesture, self._current_gesture_raw,
                activation_progress, activation_metrics,
            )
            cv2.imshow('arm_teleop_tracker', ui)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('c'), ord('C')) and visible:
                if activation_metrics and self._is_activation_pose(activation_metrics):
                    self._reference = self._dict_to_tuple(raw)
                    self._filtered = None
                    self._activation_hold = 0
                    self._set_teleop_active(True)
                    self.get_logger().info('Manual activation at vertical pose')
                else:
                    self.get_logger().warn('请先竖臂伸直朝上，再按 C')
            elif key in (ord('r'), ord('R')):
                self._reference = None
                self._filtered = None
                self._activation_hold = 0
                self._set_teleop_active(False)
                self.get_logger().info('Activation reset; raise arm straight up again')
            elif key == ord('1'):
                self._switch_side('right')
            elif key == ord('2'):
                self._switch_side('left')
            elif key in (ord('q'), ord('Q')):
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        self._cap.release()
        cv2.destroyAllWindows()
        self._landmarker.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmTeleopTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
