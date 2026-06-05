#!/usr/bin/env python3
"""PC webcam gesture recognition using MediaPipe Hands."""

import time

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GestureRecognizer(Node):
    """Detect discrete hand gestures and publish high-level robot commands."""

    GESTURE_COMMANDS = {
        'Open_Palm': 'HOME',
        'Closed_Fist': 'GRASP',
        'Pointing_Up': 'PLACE',
        'Point_Left': 'GOTO_LEFT',
        'Point_Right': 'GOTO_RIGHT',
        'Thumb_Down': 'ESTOP',
    }

    def __init__(self) -> None:
        super().__init__('gesture_recognizer')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('debounce_frames', 8)
        self.declare_parameter('publish_debug_image', True)

        self._camera_id = self.get_parameter('camera_id').value
        self._debounce_frames = int(self.get_parameter('debounce_frames').value)
        self._publish_debug = self.get_parameter('publish_debug_image').value

        self._publisher = self.create_publisher(String, 'gesture_cmd', 10)
        self._last_gesture = 'None'
        self._stable_count = 0

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._drawer = mp.solutions.drawing_utils

        self._cap = cv2.VideoCapture(self._camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_parameter('frame_width').value)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_parameter('frame_height').value)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open PC camera id={self._camera_id}')

        self._timer = self.create_timer(0.05, self._process_frame)
        self.get_logger().info('Gesture recognizer started. Show hand to PC camera.')

    @staticmethod
    def _finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def _classify_gesture(self, landmarks) -> str:
        index_up = self._finger_extended(landmarks, 8, 6)
        middle_up = self._finger_extended(landmarks, 12, 10)
        ring_up = self._finger_extended(landmarks, 16, 14)
        pinky_up = self._finger_extended(landmarks, 20, 18)

        # Thumb uses x-axis comparison for right hand in mirrored view.
        thumb_up = landmarks[4].x < landmarks[3].x

        extended = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        extended_count = sum(extended)

        wrist_x = landmarks[0].x
        index_tip_x = landmarks[8].x

        if extended_count >= 4:
            return 'Open_Palm'
        if extended_count == 0:
            return 'Closed_Fist'
        if index_up and not middle_up and not ring_up and not pinky_up:
            if index_tip_x < wrist_x - 0.05:
                return 'Point_Left'
            if index_tip_x > wrist_x + 0.05:
                return 'Point_Right'
            return 'Pointing_Up'
        if index_up and middle_up and not ring_up and not pinky_up:
            return 'Pointing_Up'
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return 'Thumb_Down'
        return 'None'

    def _publish_if_stable(self, gesture: str) -> None:
        if gesture == self._last_gesture:
            self._stable_count += 1
        else:
            self._last_gesture = gesture
            self._stable_count = 1

        if self._stable_count != self._debounce_frames:
            return
        if gesture == 'None':
            return

        command = self.GESTURE_COMMANDS.get(gesture)
        if command is None:
            return

        msg = String()
        msg.data = command
        self._publisher.publish(msg)
        self.get_logger().info(f'Gesture={gesture} -> Command={command}')
        self._stable_count = 0
        self._last_gesture = 'None'

    def _process_frame(self) -> None:
        ok, frame = self._cap.read()
        if not ok:
            self.get_logger().warn('Failed to read PC camera frame')
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        gesture = 'None'

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            gesture = self._classify_gesture(hand.landmark)
            self._drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            self._publish_if_stable(gesture)

        if self._publish_debug:
            cv2.putText(
                frame,
                f'Gesture: {gesture}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow('gesture_recognizer', frame)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        self._cap.release()
        cv2.destroyAllWindows()
        self._hands.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GestureRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
