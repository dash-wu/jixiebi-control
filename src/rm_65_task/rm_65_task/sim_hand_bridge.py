#!/usr/bin/env python3
"""Bridge RealMan dexterous-hand topics to simulated Allegro Hand."""

import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Handangle

from rm_65_task.sim_hand_client import SimHandClient


class SimHandBridge(Node):
    """Expose /rm_driver/set_hand_angle_cmd for Allegro Hand simulation."""

    def __init__(self) -> None:
        super().__init__('sim_hand_bridge')
        self.declare_parameter('enabled', True)
        enabled = bool(self.get_parameter('enabled').value)
        self._hand = SimHandClient(self, enabled=enabled)
        self.create_subscription(
            Handangle,
            '/rm_driver/set_hand_angle_cmd',
            self._on_hand_angle,
            10,
        )
        self.create_subscription(
            Handangle,
            '/rm_driver/set_hand_follow_angle_cmd',
            self._on_hand_angle,
            10,
        )
        self._hand.open()
        self.get_logger().info(
            'Allegro Hand sim bridge ready (PAL Robotics, Apache-2.0). '
            'GRASP/RELEASE gestures or Handangle topics will move the hand.'
        )

    def _on_hand_angle(self, msg: Handangle) -> None:
        self._hand.set_hand_angle(msg.hand_angle)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimHandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
