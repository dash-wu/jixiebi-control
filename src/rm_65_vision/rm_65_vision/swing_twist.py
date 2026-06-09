#!/usr/bin/env python3
"""Swing-twist decomposition of upper-arm direction for J1/J2 decoupling."""

import math
from typing import Tuple

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return vec.copy()
    return vec / n


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation."""
    axis_u = _normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vec * cos_a
        + np.cross(axis_u, vec) * sin_a
        + axis_u * np.dot(axis_u, vec) * (1.0 - cos_a)
    )


def signed_angle_about_axis(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from v1 to v2 around axis."""
    axis_u = _normalize(axis)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    v1_u = v1 / n1
    v2_u = v2 / n2
    cross = np.cross(v1_u, v2_u)
    return float(math.atan2(np.dot(cross, axis_u), np.dot(v1_u, v2_u)))


def upper_arm_elevation_rad(upper_arm: np.ndarray, down: np.ndarray) -> float:
    """0=down, pi/2=horizontal, pi=straight up."""
    u = _normalize(upper_arm)
    down_u = _normalize(down)
    vertical = -float(np.dot(u, down_u))
    horizontal = float(np.linalg.norm(u - np.dot(u, down_u) * down_u))
    return float(math.atan2(vertical, horizontal + 1e-8) + math.pi / 2)


def swing_twist_upper_arm(
    ref_upper: np.ndarray,
    cur_upper: np.ndarray,
    twist_axis: np.ndarray,
) -> Tuple[float, float, float]:
    """Decompose upper-arm motion into yaw (twist) and elevation (swing).

    Returns:
        yaw_rad: rotation about twist_axis (J1)
        elev_rad: absolute elevation after removing twist (J2 source)
        horizontal_norm: norm of horizontal projection (for J1 gating)
    """
    down = _normalize(twist_axis)
    ref_u = _normalize(ref_upper)
    cur_u = _normalize(cur_upper)

    ref_h = ref_u - np.dot(ref_u, down) * down
    cur_h = cur_u - np.dot(cur_u, down) * down
    rh = float(np.linalg.norm(ref_h))
    ch = float(np.linalg.norm(cur_h))
    h_norm = ch

    if rh < 1e-3 or ch < 1e-3:
        yaw = 0.0
        cur_detwist = cur_u
    else:
        yaw = signed_angle_about_axis(ref_h / rh, cur_h / ch, down)
        cur_detwist = rotate_vector(cur_u, down, -yaw)

    elev = upper_arm_elevation_rad(cur_detwist, down)
    return yaw, elev, h_norm
