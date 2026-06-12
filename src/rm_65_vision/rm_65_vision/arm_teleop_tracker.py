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
from rm_65_vision.joint_filter import LandmarkFilter3D
from rm_65_vision.swing_twist import swing_twist_upper_arm, upper_arm_elevation_rad
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

    ARM_JOINT_NAMES = [
        'upper_arm_yaw',
        'upper_arm_elev',
        'elbow_bend',
        'wrist_pitch',
    ]

    JOINT_NAMES = ARM_JOINT_NAMES + ['palm_open']

    JOINT_LABELS = {
        'upper_arm_yaw': '大臂左右',
        'upper_arm_elev': '大臂抬升',
        'elbow_bend': '肘部弯曲',
        'wrist_pitch': '手腕俯仰',
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
        self.declare_parameter('mirror_preview', True)
        self.declare_parameter('auto_detect_arm', True)
        self.declare_parameter('upper_arm_vector_smoothing', 0.55)
        self.declare_parameter('upper_arm_joint_smoothing', 0.42)
        self.declare_parameter('calibration_file', '')

        self._camera_id = self.get_parameter('camera_id').value
        self._smoothing = float(self.get_parameter('smoothing').value)
        self._show_debug = self.get_parameter('show_debug').value
        self._publish_ros = self.get_parameter('publish_ros').value
        self._min_visibility = float(self.get_parameter('min_visibility').value)
        self._mirror_preview = bool(self.get_parameter('mirror_preview').value)
        self._upper_arm_vector_smoothing = float(
            self.get_parameter('upper_arm_vector_smoothing').value
        )
        self._upper_arm_joint_smoothing = float(
            self.get_parameter('upper_arm_joint_smoothing').value
        )

        side = self.get_parameter('tracking_side').value.lower()
        self._auto_detect_arm = bool(self.get_parameter('auto_detect_arm').value)
        self._manual_side_lock = False
        self._auto_side_hold = 0
        self._apply_tracking_side(side)

        self._publisher = self.create_publisher(JointState, 'human_arm_joints', 10)
        self._gesture_pub = self.create_publisher(String, 'hand_gesture', 10)
        self._gesture_cmd_pub = self.create_publisher(String, 'gesture_cmd', 10)
        self._status_pub = self.create_publisher(String, 'teleop_status', 10)
        self._activation_cfg = self._load_activation_config()
        self._tracking_cfg = self._load_tracking_config()
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
        self._ref_torso: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._ref_upper_arm_dir: Optional[np.ndarray] = None
        self._use_world_landmarks = False
        self._world_warned = False
        self._filtered: Optional[Tuple[float, ...]] = None
        self._last_raw: Optional[Dict[str, float]] = None
        self._prev_elbow_delta: Optional[float] = None
        self._last_yaw_delta: Optional[float] = None
        self._calibration_active = False
        self._calibration_step = 0
        self._calibration_samples: list = []
        oe = self._tracking_cfg.get('one_euro', {})
        self._landmark_filter = LandmarkFilter3D(
            min_cutoff=float(oe.get('min_cutoff', 1.0)),
            beta=float(oe.get('beta', 0.007)),
            d_cutoff=float(oe.get('d_cutoff', 1.0)),
        )
        self._last_horizontal_norm = 0.0
        self._body_angle_deg = 0.0
        self._body_facing_camera = False
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
            'Raise arm straight up to activate follow. Keys: '
            '[R]=reset [K]=calibrate [1]=right [2]=left [Q]=quit'
        )

    def _load_tracking_config(self) -> dict:
        config_file = self.get_parameter('config_file').value
        defaults = {
            'j1_min_horizontal_norm': 0.05,
            'j1_yaw_hold': True,
            'one_euro': {'min_cutoff': 1.0, 'beta': 0.007, 'd_cutoff': 1.0},
        }
        if not config_file:
            return defaults
        with open(config_file, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        tracking = data.get('tracking', {})
        merged = {**defaults, **tracking}
        if 'one_euro' in tracking:
            merged['one_euro'] = {**defaults['one_euro'], **tracking['one_euro']}
        return merged

    CALIBRATION_STEPS = [
        ('竖臂朝上', {'upper_arm_yaw': 0.0, 'upper_arm_elev': 0.0}),
        ('大臂左/右摆约45°', {'upper_arm_yaw': 0.785, 'upper_arm_elev': 0.0}),
        ('大臂前伸至水平', {'upper_arm_yaw': 0.0, 'upper_arm_elev': -0.785}),
    ]

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
    def _elbow_flexion_rad(shoulder, elbow, wrist) -> float:
        """Elbow flexion: 0=straight, pi/2=90deg bend. Uses triangle law of cosines."""
        s = ArmTeleopTracker._lm_to_vec(shoulder)
        e = ArmTeleopTracker._lm_to_vec(elbow)
        w = ArmTeleopTracker._lm_to_vec(wrist)
        a = float(np.linalg.norm(e - s))
        b = float(np.linalg.norm(w - e))
        c = float(np.linalg.norm(w - s))
        if a < 1e-6 or b < 1e-6:
            return 0.0
        cos_val = (a * a + b * b - c * c) / (2.0 * a * b)
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        interior = math.acos(cos_val)
        return float(math.pi - interior)

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.acos(cos_val))

    @staticmethod
    def _signed_angle_between(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
        """Signed angle from v1 to v2 around axis (right-hand rule)."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        na = np.linalg.norm(axis)
        if n1 < 1e-8 or n2 < 1e-8 or na < 1e-8:
            return 0.0
        v1_u = v1 / n1
        v2_u = v2 / n2
        axis_u = axis / na
        cross = np.cross(v1_u, v2_u)
        return float(math.atan2(np.dot(cross, axis_u), np.dot(v1_u, v2_u)))

    def _torso_basis(self, pose_landmarks) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Body frame from pose landmarks (world or image coords)."""
        ls, rs = pose_landmarks[11], pose_landmarks[12]
        lh, rh = pose_landmarks[23], pose_landmarks[24]
        if not self._use_world_landmarks:
            min_vis = min(
                getattr(lm, 'visibility', 1.0) for lm in (ls, rs, lh, rh)
            )
            if min_vis < self._min_visibility:
                return None

        lateral_raw = self._lm_to_vec(rs) - self._lm_to_vec(ls)
        lat_norm = np.linalg.norm(lateral_raw)
        if lat_norm < 1e-6:
            return None
        lateral = lateral_raw / lat_norm

        mid_sh = (self._lm_to_vec(ls) + self._lm_to_vec(rs)) * 0.5
        mid_hip = (self._lm_to_vec(lh) + self._lm_to_vec(rh)) * 0.5
        down_raw = mid_hip - mid_sh
        down_norm = np.linalg.norm(down_raw)
        if down_norm < 1e-6:
            return None
        down = down_raw / down_norm

        forward = np.cross(lateral, down)
        fwd_norm = np.linalg.norm(forward)
        if fwd_norm < 1e-6:
            return None
        forward = forward / fwd_norm
        return lateral, down, forward

    def _resolve_pose_for_angles(self, result, pose_landmarks):
        """Prefer world landmarks for joint angles; image landmarks for UI."""
        world = getattr(result, 'pose_world_landmarks', None)
        if world and len(world) >= 17:
            if not self._use_world_landmarks:
                self._use_world_landmarks = True
                self.get_logger().info('Using pose_world_landmarks for joint angles')
            return world
        if not self._world_warned:
            self._world_warned = True
            self.get_logger().warn(
                'pose_world_landmarks unavailable; falling back to image coords'
            )
        self._use_world_landmarks = False
        return pose_landmarks

    def _update_body_orientation(self, image_pose_landmarks) -> None:
        """Estimate how much the user is facing the camera from shoulder line.

        Uses image-space shoulders: slope of the shoulder line.
        - Near 0° = facing camera (bad for J1 depth estimation)
        - 20°~50° = side view (good)
        - Near 90° = profile (arm occluded)

        Also detects head/face yaw from nose-ear landmarks for cross-check.
        """
        ls = image_pose_landmarks[11]
        rs = image_pose_landmarks[12]
        # Shoulder-line angle in image plane w.r.t. horizontal
        dx = float(rs.x - ls.x)
        dy = float(rs.y - ls.y)
        if abs(dx) < 0.005:
            self._body_angle_deg = 90.0
            self._body_facing_camera = False
            return
        slope_deg = math.degrees(math.atan2(dy, dx))
        # Lower slope → shoulders nearly horizontal → facing camera
        self._body_angle_deg = abs(slope_deg)

        # Also check nose-to-ear ratio as cross-check for head yaw
        nose = image_pose_landmarks[0]
        le, re = image_pose_landmarks[7], image_pose_landmarks[8]
        ear_mid = (le.x + re.x) * 0.5
        nose_offset_norm = abs(nose.x - ear_mid) / max(abs(re.x - le.x), 0.01)
        # nose_offset ≈ 0.5 when facing camera, large when turned
        head_on = nose_offset_norm < 0.8

        cfg = self._tracking_cfg
        front_th = float(cfg.get('body_front_facing_threshold_deg', 18.0))
        self._body_facing_camera = (
            self._body_angle_deg < front_th and head_on
        )

    def _filtered_arm_points(
        self,
        shoulder,
        elbow,
        wrist,
        timestamp_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = {
            'shoulder': self._lm_to_vec(shoulder),
            'elbow': self._lm_to_vec(elbow),
            'wrist': self._lm_to_vec(wrist),
        }
        if self._use_world_landmarks:
            points = self._landmark_filter.filter(points, timestamp_sec)
        return points['shoulder'], points['elbow'], points['wrist']

    @staticmethod
    def _upper_arm_yaw_in_frame(
        upper_arm: np.ndarray,
        lateral: np.ndarray,
        down: np.ndarray,
        forward: np.ndarray,
    ) -> float:
        """Azimuth of upper arm in a fixed torso frame (0 when arm points forward-up)."""
        u_h = upper_arm - np.dot(upper_arm, down) * down
        h_norm = np.linalg.norm(u_h)
        if h_norm < 1e-3:
            return 0.0
        u_h /= h_norm
        sin_yaw = float(np.dot(np.cross(forward, u_h), down))
        cos_yaw = float(np.dot(forward, u_h))
        return float(math.atan2(sin_yaw, cos_yaw))

    @staticmethod
    def _upper_arm_yaw_rad(upper_arm: np.ndarray, lateral: np.ndarray, down: np.ndarray) -> float:
        forward = np.cross(lateral, down)
        fwd_norm = np.linalg.norm(forward)
        if fwd_norm < 1e-6:
            return float(math.atan2(upper_arm[0], upper_arm[2] + 1e-8))
        forward /= fwd_norm
        return ArmTeleopTracker._upper_arm_yaw_in_frame(upper_arm, lateral, down, forward)

    @staticmethod
    def _upper_arm_elevation_rad(upper_arm: np.ndarray, down: Optional[np.ndarray] = None) -> float:
        """Elevation: 0=down, pi/2=horizontal, pi=straight up.

        atan2 form keeps good sensitivity near vertical (activation pose).
        """
        u_norm = np.linalg.norm(upper_arm)
        if u_norm < 1e-8:
            return math.pi / 2
        u = upper_arm / u_norm
        if down is not None:
            vertical = -float(np.dot(u, down))
            horizontal = float(np.linalg.norm(u - np.dot(u, down) * down))
            return float(math.atan2(vertical, horizontal + 1e-8) + math.pi / 2)
        sideward = abs(float(upper_arm[0]))
        upward = max(-float(upper_arm[1]), 0.0)
        if sideward < 1e-8 and upward < 1e-8:
            return math.pi / 2
        return math.atan2(upward, sideward) + math.pi / 2

    def _wrist_pitch_rad(
        self,
        upper_arm: np.ndarray,
        forearm: np.ndarray,
        hand_dir: np.ndarray,
        torso: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> float:
        """Wrist flexion in 3D, referenced to torso down (not image-plane cross)."""
        f_norm = np.linalg.norm(forearm)
        if f_norm < 1e-8:
            return 0.0
        f = forearm / f_norm

        lateral = down = None
        if torso is not None:
            lateral, down, _ = torso

        ref_down = down if down is not None else np.array([0.0, 1.0, 0.0])
        flex_ref = ref_down - np.dot(ref_down, f) * f
        flex_norm = np.linalg.norm(flex_ref)
        if flex_norm < 1e-6 and lateral is not None:
            flex_ref = lateral - np.dot(lateral, f) * f
            flex_norm = np.linalg.norm(flex_ref)

        if flex_norm < 1e-6:
            arm_normal = np.cross(upper_arm, forearm)
            n_norm = np.linalg.norm(arm_normal)
            if n_norm < 1e-8:
                return self._angle_between(forearm, hand_dir)
            arm_normal /= n_norm
            if lateral is not None and np.dot(arm_normal, lateral) < 0.0:
                arm_normal = -arm_normal
            return self._signed_angle_between(forearm, hand_dir, arm_normal)

        flex_ref /= flex_norm
        axis = np.cross(f, flex_ref)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return self._angle_between(forearm, hand_dir)
        axis /= axis_norm
        if lateral is not None and np.dot(axis, lateral) < 0.0:
            axis = -axis
        return self._signed_angle_between(forearm, hand_dir, axis)

    def _mp_side_for_user(self, user_side: str) -> str:
        """Map user's anatomical side to MediaPipe left/right (mirrored image swaps them)."""
        if self._mirror_preview:
            return 'left' if user_side == 'right' else 'right'
        return user_side

    def _arm_ids_from_user_side(self, user_side: str) -> dict:
        mp_side = self._mp_side_for_user(user_side)
        return self.RIGHT_ARM if mp_side == 'right' else self.LEFT_ARM

    def _apply_tracking_side(self, user_side: str) -> None:
        self._tracking_side = user_side
        self._arm_ids = self._arm_ids_from_user_side(user_side)

    def _arm_activity_score(self, pose_landmarks, user_side: str) -> float:
        ids = self._arm_ids_from_user_side(user_side)
        vis = [
            getattr(pose_landmarks[ids[name]], 'visibility', 1.0)
            for name in ('shoulder', 'elbow', 'wrist')
        ]
        if min(vis) < self._min_visibility:
            return 0.0
        shoulder = pose_landmarks[ids['shoulder']]
        elbow = pose_landmarks[ids['elbow']]
        wrist = pose_landmarks[ids['wrist']]
        raise_amt = max(float(shoulder.y - elbow.y), 0.0) + max(float(elbow.y - wrist.y), 0.0)
        return min(vis) * (1.0 + raise_amt * 8.0)

    def _maybe_auto_detect_arm(self, pose_landmarks) -> None:
        if not self._auto_detect_arm or self._teleop_active or self._manual_side_lock:
            return
        left_score = self._arm_activity_score(pose_landmarks, 'left')
        right_score = self._arm_activity_score(pose_landmarks, 'right')
        if max(left_score, right_score) < 0.05:
            self._auto_side_hold = 0
            return
        detected = 'left' if left_score >= right_score else 'right'
        if detected == self._tracking_side:
            self._auto_side_hold = 0
            return
        self._auto_side_hold += 1
        if self._auto_side_hold >= 10:
            self._apply_tracking_side(detected)
            self._reset_reference_state()
            self._activation_hold = 0
            self._auto_side_hold = 0
            label = '左臂' if detected == 'left' else '右臂'
            self.get_logger().info(f'Auto-detected active arm: {label}')

    def _visible(self, pose_landmarks, name: str) -> bool:
        idx = self._arm_ids[name]
        lm = pose_landmarks[idx]
        return getattr(lm, 'visibility', 1.0) >= self._min_visibility

    def _get_hand_landmarks(self, result):
        """Return 21-point hand landmarks for the tracked arm side, if valid."""
        mp_side = self._mp_side_for_user(self._tracking_side)
        hand = result.right_hand_landmarks if mp_side == 'right' else result.left_hand_landmarks
        if GestureClassifier.is_valid(hand):
            return hand
        # 追踪侧未检测到时，尝试另一只手（侧身时可能只检测到一只）
        fallback = result.left_hand_landmarks if mp_side == 'right' else result.right_hand_landmarks
        if GestureClassifier.is_valid(fallback):
            return fallback
        return None

    def _reset_motion_filters(self) -> None:
        self._prev_elbow_delta = None
        self._last_yaw_delta = None
        self._landmark_filter.reset()

    def _reset_reference_state(self) -> None:
        self._reference = None
        self._ref_torso = None
        self._ref_upper_arm_dir = None
        self._filtered = None
        self._reset_motion_filters()

    def _capture_activation_reference(
        self,
        raw: Dict[str, float],
        angle_pose_landmarks,
        upper_arm_raw: np.ndarray,
    ) -> None:
        torso = self._torso_basis(angle_pose_landmarks)
        ref_values = dict(raw)
        if torso is not None:
            self._ref_torso = tuple(axis.copy() for axis in torso)
            u_norm = float(np.linalg.norm(upper_arm_raw))
            if u_norm > 1e-6:
                self._ref_upper_arm_dir = upper_arm_raw / u_norm
            yaw, elev, h_norm = self._compute_upper_arm_joints(upper_arm_raw)
            ref_values['upper_arm_yaw'] = yaw
            ref_values['upper_arm_elev'] = elev
            self._last_horizontal_norm = h_norm
            self._last_yaw_delta = 0.0
        self._reference = self._dict_to_tuple(ref_values)

    def _compute_upper_arm_joints(
        self,
        upper_arm_raw: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Swing-twist: yaw (J1), elev (J2), horizontal norm for gating."""
        if self._ref_torso is None or self._ref_upper_arm_dir is None:
            return 0.0, upper_arm_elevation_rad(upper_arm_raw, np.array([0.0, 1.0, 0.0])), 0.0
        _, down, _ = self._ref_torso
        yaw, elev, h_norm = swing_twist_upper_arm(
            self._ref_upper_arm_dir, upper_arm_raw, down
        )
        return yaw, elev, h_norm

    def _compute_arm_angles(
        self,
        angle_pose_landmarks,
        hand_landmarks,
        palm_open: float,
        timestamp_sec: float,
    ) -> Dict[str, float]:
        shoulder = angle_pose_landmarks[self._arm_ids['shoulder']]
        elbow = angle_pose_landmarks[self._arm_ids['elbow']]
        wrist = angle_pose_landmarks[self._arm_ids['wrist']]

        sh_vec, el_vec, wr_vec = self._filtered_arm_points(
            shoulder, elbow, wrist, timestamp_sec
        )
        upper_arm_raw = el_vec - sh_vec
        forearm_raw = wr_vec - el_vec

        torso = self._torso_basis(angle_pose_landmarks)

        if self._ref_torso is not None and self._ref_upper_arm_dir is not None:
            upper_arm_yaw, upper_arm_elev, h_norm = self._compute_upper_arm_joints(
                upper_arm_raw
            )
            self._last_horizontal_norm = h_norm
        elif torso is not None:
            _, down, _ = torso
            upper_arm_yaw = 0.0
            upper_arm_elev = upper_arm_elevation_rad(upper_arm_raw, down)
        else:
            upper_arm_yaw = float(math.atan2(upper_arm_raw[0], upper_arm_raw[2] + 1e-8))
            upper_arm_elev = upper_arm_elevation_rad(upper_arm_raw)

        elbow_bend = self._elbow_flexion_from_vec(sh_vec, el_vec, wr_vec)

        hand_dir = forearm_raw.copy()
        if GestureClassifier.is_valid(hand_landmarks):
            wrist_h = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            hand_dir = self._lm_to_vec(middle_tip) - self._lm_to_vec(wrist_h)

        wrist_pitch = self._wrist_pitch_rad(
            upper_arm_raw, forearm_raw, hand_dir, torso
        )

        return {
            'upper_arm_yaw': upper_arm_yaw,
            'upper_arm_elev': upper_arm_elev,
            'elbow_bend': elbow_bend,
            'wrist_pitch': wrist_pitch,
            'palm_open': palm_open,
        }

    @staticmethod
    def _elbow_flexion_from_vec(
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> float:
        a = float(np.linalg.norm(elbow - shoulder))
        b = float(np.linalg.norm(wrist - elbow))
        c = float(np.linalg.norm(wrist - shoulder))
        if a < 1e-6 or b < 1e-6:
            return 0.0
        cos_val = (a * a + b * b - c * c) / (2.0 * a * b)
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        return float(math.pi - math.acos(cos_val))

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

    def _apply_j1_yaw_hold(self, values: Dict[str, float], horizontal_norm: float) -> Dict[str, float]:
        """Hold J1 delta when upper arm is near-vertical (gimbal singularity).

        Also more aggressive when body is facing camera — depth estimation
        is unreliable so J1 should only follow when user is clearly turned.
        """
        if not self._tracking_cfg.get('j1_yaw_hold', True):
            return values
        min_h = float(self._tracking_cfg.get('j1_min_horizontal_norm', 0.05))
        # When front-facing, require higher horizontal norm to trust J1
        if self._body_facing_camera:
            min_h = float(self._tracking_cfg.get('j1_min_horizontal_norm_front', 0.15))
        if horizontal_norm >= min_h:
            self._last_yaw_delta = float(values['upper_arm_yaw'])
            return values
        if self._last_yaw_delta is not None:
            result = dict(values)
            result['upper_arm_yaw'] = self._last_yaw_delta
            return result
        return values

    def _apply_reference(self, values: Dict[str, float]) -> Dict[str, float]:
        if self._reference is None:
            return values
        ref = self._tuple_to_dict(self._reference)
        angular = {'upper_arm_yaw', 'wrist_pitch'}
        result = {}
        for key in self.JOINT_NAMES:
            delta = values[key] - ref[key]
            if key in angular:
                delta = math.atan2(math.sin(delta), math.cos(delta))
            result[key] = delta
        return result

    def _decouple_deltas(self, values: Dict[str, float]) -> Dict[str, float]:
        """Suppress J1 only while elbow is actively moving (not when elbow stays bent)."""
        elbow_delta = abs(float(values.get('elbow_bend', 0.0)))
        yaw_delta = abs(float(values.get('upper_arm_yaw', 0.0)))

        elbow_rate = 0.0
        if self._prev_elbow_delta is not None:
            elbow_rate = abs(elbow_delta - self._prev_elbow_delta)
        self._prev_elbow_delta = elbow_delta

        if elbow_rate < 0.012 or elbow_rate <= yaw_delta * 0.6:
            return values

        strength = min(1.0, (elbow_rate - 0.012) / 0.06)
        result = dict(values)
        result['upper_arm_yaw'] = float(values['upper_arm_yaw']) * (1.0 - 0.75 * strength)
        return result

    def _smooth(self, values: Dict[str, float]) -> Dict[str, float]:
        tup = self._dict_to_tuple(values)
        if self._filtered is None:
            self._filtered = tup
            return values
        alpha_by_joint = {
            'upper_arm_yaw': self._upper_arm_joint_smoothing,
            'upper_arm_elev': 0.42,
            'elbow_bend': 0.12,
            'wrist_pitch': self._smoothing,
            'palm_open': self._smoothing,
        }
        alphas = [alpha_by_joint.get(name, self._smoothing) for name in self.JOINT_NAMES]
        smoothed = tuple(
            alpha * v + (1.0 - alpha) * f
            for alpha, v, f in zip(alphas, tup, self._filtered)
        )
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

        coord_cn = '世界坐标' if self._use_world_landmarks else '图像坐标'
        lines.append((f'角度源: {coord_cn}', y_start, 14, (140, 200, 255)))
        y_start += 22

        # Body orientation and J1 quality metrics
        body_label = f'身体角: {self._body_angle_deg:.0f}°'
        if self._body_facing_camera:
            body_color = (80, 120, 255)
            body_hint = ' ⚠正对镜头，J1不可靠'
        else:
            body_color = (0, 220, 120)
            body_hint = ' ✓侧面，追踪良好'
        lines.append((body_label + body_hint, y_start, 14, body_color))
        y_start += 18

        h_norm = self._last_horizontal_norm
        h_label = f'水平模: {h_norm:.3f}'
        if h_norm < 0.05:
            h_color = (80, 80, 255)
        elif h_norm < 0.10:
            h_color = (255, 220, 80)
        else:
            h_color = (0, 220, 120)
        if self._body_facing_camera:
            lines.append((f'{h_label} (J1锁定中)', y_start, 14, (80, 120, 255)))
        else:
            lines.append((f'{h_label} ({">0.15" if self._body_facing_camera else ">0.05"}→J1解锁)', y_start, 14, h_color))
        y_start += 22

        if self._body_facing_camera:
            lines.append(('⚠ 请侧身30~45°面对镜头', y_start, 14, (0, 180, 255)))
            y_start += 20
        if self._calibration_active:
            label, _ = self.CALIBRATION_STEPS[self._calibration_step]
            lines.append((f'标定 [{self._calibration_step + 1}/3]: {label}', y_start, 15, (0, 255, 200)))
            lines.append(('按 K 确认当前姿势', y_start + 20, 14, (180, 180, 180)))
            y_start += 44

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
        for name in self.ARM_JOINT_NAMES:
            val = values[name]
            if name == 'elbow_bend' and self._last_raw is not None:
                abs_deg = math.degrees(self._last_raw['elbow_bend'])
                text = (
                    f'{self.JOINT_LABELS[name]}: Δ{math.degrees(val):+.0f}° '
                    f'(实际{abs_deg:.0f}°)'
                )
            elif name == 'upper_arm_yaw' and self._last_raw is not None:
                abs_deg = math.degrees(self._last_raw['upper_arm_yaw'])
                text = (
                    f'{self.JOINT_LABELS[name]}: Δ{math.degrees(val):+.0f}° '
                    f'(方位{abs_deg:+.0f}°)'
                )
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
        for hint in ['竖臂朝上自动激活', 'R  重置', 'K  三点标定', '1/2  锁定右/左臂', 'Q  退出']:
            lines.append((hint, y, 15, (120, 120, 120)))
            y += 20

        panel = render_panel(self.PANEL_WIDTH, canvas_h, lines)
        canvas[:canvas_h, w:w + self.PANEL_WIDTH] = panel
        for i, (_label, color) in enumerate(legend):
            ly = legend_y_start + i * 24
            cv2.line(canvas, (w + 16, ly - 4), (w + 44, ly - 4), color, 4, cv2.LINE_AA)
        return canvas

    def _update_gesture(self, hand_landmarks) -> None:
        is_right = self._mp_side_for_user(self._tracking_side) == 'right'
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
        parts = [f'{k}={math.degrees(self._last_raw[k]):.1f}' for k in self.ARM_JOINT_NAMES]
        parts.append(f'palm={self._last_raw["palm_open"]:.2f}')
        parts.append(f'gesture={GestureClassifier.label_cn(self._current_gesture)}')
        self.get_logger().info(f'Detection {rate:.0f}% | ' + ' '.join(parts))

    def _switch_side(self, side: str) -> None:
        self._manual_side_lock = True
        self._auto_side_hold = 0
        self._apply_tracking_side(side)
        self._reset_reference_state()
        self._activation_hold = 0
        self._teleop_active = False
        self._publish_teleop_status('WAITING')
        self._gesture_debouncer = GestureDebouncer(
            int(self.get_parameter('gesture_debounce_frames').value)
        )
        self._current_gesture = 'None'
        label = '左臂' if side == 'left' else '右臂'
        self.get_logger().info(f'Manual switch -> {label}')

    def _calibration_file_path(self) -> str:
        cal = self.get_parameter('calibration_file').value
        if cal:
            return str(cal)
        try:
            task_share = get_package_share_directory('rm_65_task')
            return os.path.join(task_share, 'config', 'arm_teleop_calibration.yaml')
        except Exception:
            return os.path.expanduser('~/ros2_ws/src/rm_65_task/config/arm_teleop_calibration.yaml')

    def _start_calibration(self) -> None:
        self._calibration_active = True
        self._calibration_step = 0
        self._calibration_samples = []
        self.get_logger().info(
            f'Calibration started: step 1/3 -> {self.CALIBRATION_STEPS[0][0]}'
        )

    def _finish_calibration(self) -> None:
        if len(self._calibration_samples) < 3:
            self.get_logger().warn('Calibration incomplete; need 3 poses')
            return
        ref = self._calibration_samples[0]['measured']
        scales: Dict[str, float] = {}
        for key in ('upper_arm_yaw', 'upper_arm_elev', 'elbow_bend'):
            measured: list = []
            expected: list = []
            for sample in self._calibration_samples[1:]:
                m = float(sample['measured'][key]) - float(ref.get(key, 0.0))
                e = float(sample['expected'][key])
                if abs(m) > 0.02:
                    measured.append(m)
                    expected.append(e)
            if not measured:
                scales[key] = 1.0
                continue
            num = sum(e * m for e, m in zip(expected, measured))
            den = sum(m * m for m in measured)
            scale = num / den if den > 1e-6 else 1.0
            scales[key] = float(max(0.3, min(3.0, scale)))

        out_path = self._calibration_file_path()
        payload = {
            'calibration': {
                'upper_arm_yaw': {'scale': scales.get('upper_arm_yaw', 1.0), 'offset': 0.0},
                'upper_arm_elev': {'scale': scales.get('upper_arm_elev', 1.0), 'offset': 0.0},
                'elbow_bend': {'scale': scales.get('elbow_bend', 1.0), 'offset': 0.0},
                'wrist_pitch': {'scale': 1.0, 'offset': 0.0},
            }
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
        self._calibration_active = False
        self._calibration_step = 0
        self.get_logger().info(f'Calibration saved -> {out_path}')

    def _calibration_capture(self, raw: Dict[str, float]) -> None:
        if not self._reference:
            self.get_logger().warn('请先激活跟随，再按 K 开始标定')
            return
        deltas = self._apply_reference(raw)
        label, expected = self.CALIBRATION_STEPS[self._calibration_step]
        self._calibration_samples.append({
            'label': label,
            'expected': expected,
            'measured': dict(deltas),
        })
        self.get_logger().info(
            f'Calibration captured {self._calibration_step + 1}/3: {label}'
        )
        self._calibration_step += 1
        if self._calibration_step >= len(self.CALIBRATION_STEPS):
            self._finish_calibration()
        else:
            next_label, _ = self.CALIBRATION_STEPS[self._calibration_step]
            self.get_logger().info(f'Next pose: {next_label}')

    def _process_frame(self) -> None:
        ok, frame = self._cap.read()
        self._frames_total += 1
        if not ok:
            self.get_logger().warn('Failed to read PC camera frame')
            return

        if self._mirror_preview:
            frame = cv2.flip(frame, 1)

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
        self._update_body_orientation(pose_landmarks)
        angle_pose_landmarks = self._resolve_pose_for_angles(result, pose_landmarks)
        timestamp_sec = self._timestamp_ms / 1000.0
        self._maybe_auto_detect_arm(pose_landmarks)
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
            is_right_mp = self._mp_side_for_user(self._tracking_side) == 'right'
            palm_open = self._palm_openness(hand_landmarks, is_right_mp)

        self._update_gesture(hand_landmarks)

        raw = self._compute_arm_angles(
            angle_pose_landmarks, hand_landmarks, palm_open, timestamp_sec
        )
        self._last_raw = raw.copy()

        upper_arm_raw = (
            self._lm_to_vec(angle_pose_landmarks[self._arm_ids['elbow']])
            - self._lm_to_vec(angle_pose_landmarks[self._arm_ids['shoulder']])
        )

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
                    self._capture_activation_reference(
                        raw, angle_pose_landmarks, upper_arm_raw
                    )
                    self._filtered = None
                    self._set_teleop_active(True)
            else:
                self._activation_hold = 0

        ref_values = self._apply_j1_yaw_hold(
            self._apply_reference(raw), self._last_horizontal_norm
        )
        values = self._smooth(self._decouple_deltas(ref_values))
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
                    self._capture_activation_reference(
                        raw, angle_pose_landmarks, upper_arm_raw
                    )
                    self._filtered = None
                    self._activation_hold = 0
                    self._set_teleop_active(True)
                    self.get_logger().info('Manual activation at vertical pose')
                else:
                    self.get_logger().warn('请先竖臂伸直朝上，再按 C')
            elif key in (ord('k'), ord('K')):
                if self._calibration_active:
                    self._calibration_capture(raw)
                elif self._teleop_active:
                    self._start_calibration()
                else:
                    self.get_logger().warn('请先激活跟随，再按 K 标定')
            elif key in (ord('r'), ord('R')):
                self._calibration_active = False
                self._calibration_step = 0
                self._reset_reference_state()
                self._activation_hold = 0
                self._manual_side_lock = False
                self._auto_side_hold = 0
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
