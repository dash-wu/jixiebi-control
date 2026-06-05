#!/usr/bin/env python3
"""Map human arm joints to RM65 joint targets for grasp-oriented teleoperation."""

import math
from typing import Dict, List, Optional


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value - deadzone if value > 0 else value + deadzone


def apply_human_delta(value: float, cfg: dict) -> float:
    delta = apply_deadzone(float(value), float(cfg.get('deadzone', 0.0)))
    if cfg.get('wrap', False):
        delta = wrap_angle(delta)
    max_delta = cfg.get('max_delta')
    if max_delta is not None:
        limit = float(max_delta)
        delta = max(-limit, min(limit, delta))
    return delta


class TeleopMapper:
    """Convert calibrated human arm state into RM65 joint commands."""

    HUMAN_JOINTS = [
        'upper_arm_yaw',
        'upper_arm_elev',
        'elbow_bend',
        'wrist_pitch',
        'wrist_roll',
    ]

    def __init__(self, config: dict) -> None:
        self._config = config
        home = config.get('robot_ready_pose', config.get('robot_home', [0.0] * 6))
        self._home = [float(v) for v in home]
        if len(self._home) != 6:
            raise ValueError('robot_ready_pose / robot_home must have 6 joint values')
        raw_active = config.get('active_joint_indices')
        if raw_active is None:
            self._active_indices = set(range(6))
        else:
            self._active_indices = {int(i) for i in raw_active}
        self._filtered: Optional[List[float]] = None
        self._grasp_mode = False

    def set_grasp_mode(self, enabled: bool) -> None:
        self._grasp_mode = enabled

    def go_home(self) -> List[float]:
        self._filtered = list(self._home)
        return list(self._home)

    def map_joints(self, human: Dict[str, float]) -> List[float]:
        robot = list(self._home)

        for human_name, cfg in self._config.get('mapping', {}).items():
            if human_name not in human:
                continue
            idx = int(cfg['index'])
            scale = float(cfg.get('scale', 1.0))
            deadzone = float(cfg.get('deadzone', 0.0))
            delta = apply_human_delta(float(human[human_name]), cfg)
            robot[idx] += scale * delta

        if self._grasp_mode:
            approach = self._config.get('grasp', {}).get('approach_delta', [0.0] * 6)
            for i in range(6):
                robot[i] += float(approach[i])

        robot = self._clamp(robot)
        robot = self._smooth(robot)
        return self._lock_inactive_joints(robot)

    def _lock_inactive_joints(self, robot: List[float]) -> List[float]:
        if self._active_indices == set(range(6)):
            return robot
        locked = list(robot)
        for idx in range(6):
            if idx not in self._active_indices:
                locked[idx] = self._home[idx]
        if self._filtered is not None:
            self._filtered = list(locked)
        return locked

    def retract_from_grasp(self) -> List[float]:
        self._grasp_mode = False
        if self._filtered is None:
            return list(self._home)
        retract = self._config.get('grasp', {}).get('retract_delta', [0.0] * 6)
        robot = list(self._filtered)
        for i in range(6):
            robot[i] += float(retract[i])
        robot = self._clamp(robot)
        self._filtered = robot
        return self._lock_inactive_joints(robot)

    def _clamp(self, robot: List[float]) -> List[float]:
        limits = self._config.get('joint_limits', {})
        for idx, name in enumerate(
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        ):
            if name not in limits:
                continue
            low, high = limits[name]
            robot[idx] = max(float(low), min(float(high), robot[idx]))
        return robot

    def _smooth(self, robot: List[float]) -> List[float]:
        alpha = float(self._config.get('output_smoothing', 0.25))
        if self._filtered is None:
            self._filtered = robot
            return robot
        self._filtered = [
            alpha * target + (1.0 - alpha) * current
            for target, current in zip(robot, self._filtered)
        ]
        return list(self._filtered)
