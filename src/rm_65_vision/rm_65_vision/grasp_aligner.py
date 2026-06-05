#!/usr/bin/env python3
"""Arm-mounted camera alignment using ArUco markers."""

import yaml
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


class GraspAligner(Node):
    """Detect ArUco marker in arm camera and publish pixel offset from image center."""

    def __init__(self) -> None:
        super().__init__('grasp_aligner')
        self.declare_parameter('image_topic', '/arm_camera/image_raw')
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('hand_eye_config', '')

        image_topic = self.get_parameter('image_topic').value
        dict_name = self.get_parameter('aruco_dict').value
        self._target_id = int(self.get_parameter('marker_id').value)

        self._bridge = CvBridge()
        self._dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self._parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._parameters)

        self._fx = 554.0
        self._fy = 554.0
        self._cx = 320.0
        self._cy = 240.0
        self._pixel_to_meter = 0.00015
        self._load_hand_eye()

        self._publisher = self.create_publisher(PointStamped, 'grasp_offset', 10)
        self._subscription = self.create_subscription(Image, image_topic, self._image_callback, 10)
        self.get_logger().info(f'Grasp aligner listening on {image_topic}')

    def _load_hand_eye(self) -> None:
        config_path = self.get_parameter('hand_eye_config').value
        if not config_path:
            self.get_logger().warn('Using default hand-eye parameters')
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
            self._fx = float(data.get('fx', self._fx))
            self._fy = float(data.get('fy', self._fy))
            self._cx = float(data.get('cx', self._cx))
            self._cy = float(data.get('cy', self._cy))
            self._pixel_to_meter = float(data.get('pixel_to_meter', self._pixel_to_meter))
            self.get_logger().info(f'Loaded hand-eye config: {config_path}')
        except OSError as exc:
            self.get_logger().warn(f'Failed to load hand-eye config: {exc}')

    def _image_callback(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = self._detector.detectMarkers(frame)

        if ids is None:
            return

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != self._target_id:
                continue

            center = marker_corners[0].mean(axis=0)
            u, v = float(center[0]), float(center[1])
            du = u - self._cx
            dv = v - self._cy

            offset = PointStamped()
            offset.header = msg.header
            offset.point.x = du * self._pixel_to_meter
            offset.point.y = -dv * self._pixel_to_meter
            offset.point.z = 0.0
            self._publisher.publish(offset)
            return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspAligner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
