#!/usr/bin/env python3
"""Hand gesture classification from MediaPipe hand landmarks."""

from typing import Optional, Tuple


GESTURE_LABELS_CN = {
    'None': '无',
    'Open_Palm': '张开手掌',
    'Closed_Fist': '握拳',
    'Point_Left': '食指向左',
    'Point_Right': '食指向右',
    'Pointing_Up': '食指向上',
    'Victory': '比 V',
    'Thumb_Down': '拇指向下',
}

GESTURE_COMMANDS = {
    'Open_Palm': 'RELEASE',
    'Closed_Fist': 'GRASP',
    'Pointing_Up': 'PLACE',
    'Victory': 'HOME',
    'Point_Left': 'GOTO_LEFT',
    'Point_Right': 'GOTO_RIGHT',
    'Thumb_Down': 'ESTOP',
}


class GestureClassifier:
    """Classify static hand gestures from 21 hand landmarks."""

    NUM_LANDMARKS = 21

    @classmethod
    def is_valid(cls, hand_landmarks) -> bool:
        if hand_landmarks is None:
            return False
        try:
            return len(hand_landmarks) >= cls.NUM_LANDMARKS
        except TypeError:
            return False

    @staticmethod
    def _finger_extended(hand, tip: int, pip: int) -> bool:
        return hand[tip].y < hand[pip].y

    @classmethod
    def _thumb_extended(cls, hand, is_right_arm: bool) -> bool:
        if is_right_arm:
            return hand[4].x > hand[3].x
        return hand[4].x < hand[3].x

    @classmethod
    def classify(cls, hand_landmarks, is_right_arm: bool) -> str:
        if not cls.is_valid(hand_landmarks):
            return 'None'

        index_up = cls._finger_extended(hand_landmarks, 8, 6)
        middle_up = cls._finger_extended(hand_landmarks, 12, 10)
        ring_up = cls._finger_extended(hand_landmarks, 16, 14)
        pinky_up = cls._finger_extended(hand_landmarks, 20, 18)
        thumb_up = cls._thumb_extended(hand_landmarks, is_right_arm)

        extended = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        count = sum(extended)

        wrist_x = hand_landmarks[0].x
        index_x = hand_landmarks[8].x

        if count >= 4:
            return 'Open_Palm'
        if count == 0:
            return 'Closed_Fist'
        if index_up and middle_up and not ring_up and not pinky_up:
            return 'Victory'
        if index_up and not middle_up and not ring_up and not pinky_up:
            if index_x < wrist_x - 0.04:
                return 'Point_Left'
            if index_x > wrist_x + 0.04:
                return 'Point_Right'
            return 'Pointing_Up'
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return 'Thumb_Down'
        return 'None'

    @classmethod
    def label_cn(cls, gesture: str) -> str:
        return GESTURE_LABELS_CN.get(gesture, '无')

    @classmethod
    def to_command(cls, gesture: str) -> Optional[str]:
        return GESTURE_COMMANDS.get(gesture)


class GestureDebouncer:
    """Require the same gesture for several frames before accepting it."""

    def __init__(self, debounce_frames: int = 6) -> None:
        self._debounce_frames = debounce_frames
        self._last = 'None'
        self._count = 0
        self.stable_gesture = 'None'

    def update(self, gesture: str) -> Tuple[str, bool]:
        if gesture == self._last:
            self._count += 1
        else:
            self._last = gesture
            self._count = 1

        changed = False
        if self._count >= self._debounce_frames and gesture != self.stable_gesture:
            self.stable_gesture = gesture
            changed = True
        return self.stable_gesture, changed
