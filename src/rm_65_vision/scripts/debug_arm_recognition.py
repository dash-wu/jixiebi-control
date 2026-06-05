#!/usr/bin/env python3
"""Standalone arm recognition debug without full ROS launch overhead."""

import sys

import rclpy
from rm_65_vision.arm_teleop_tracker import ArmTeleopTracker


def main() -> int:
    rclpy.init(args=None)
    node = ArmTeleopTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
