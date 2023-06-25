import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui as mouse
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class Window:
    def __init__(self):
        self.cap = None
        self.pose = mp_pose.Pose()
        self.command_key_mappings = {
            "right": "right",
            "left": "left",
            "forward": "up",
            "duck": ord(" "),
            "swing": ord("K")
        }
        self.command_active = {
            "right": False,
            "left": False,
            "forward": False,
            "duck": False,
            "swing": False
        }
        self.shoulder_width = None

    def start_capture(self):
        self.cap = cv2.VideoCapture(0)
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    break

                image_height, image_width, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                self.process_pose(results.pose_landmarks)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Pose Detection', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_pose(self, landmarks):
        if landmarks is None:
            return

        self.process_right_command(landmarks)
        self.process_left_command(landmarks)
        self.process_forward_command(landmarks)
        self.process_duck_command(landmarks)
        self.process_swing_command(landmarks)

    def process_right_command(self, landmarks):
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        distance = self.calculate_distance(right_shoulder, right_wrist)

        if distance is not None:
            if (
                    distance < 0.1 and not self.command_active["duck"] and
                    not self.command_active["forward"] and not self.command_active["left"]
            ):
                self.command_active["right"] = True
                keyboard.press(self.command_key_mappings["right"])
            elif distance >= 0.1 and self.command_active["right"]:
                self.command_active["right"] = False
                keyboard.release(self.command_key_mappings["right"])

    def process_left_command(self, landmarks):
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        distance = self.calculate_distance(left_shoulder, left_wrist)

        if distance is not None:
            if (
                    distance < 0.1 and not self.command_active["duck"] and
                    not self.command_active["forward"] and not self.command_active["right"]
            ):
                self.command_active["left"] = True
                keyboard.press(self.command_key_mappings["left"])
            elif distance >= 0.1 and self.command_active["left"]:
                self.command_active["left"] = False
                keyboard.release(self.command_key_mappings["left"])

    def process_forward_command(self, landmarks):
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        if right_wrist is not None and left_wrist is not None and right_shoulder is not None and left_shoulder is not None:
            if (
                    self.are_points_overlapping(right_wrist, left_wrist) and
                    self.is_body_leaning_forward(right_shoulder, left_shoulder) and
                    not self.command_active["right"] and not self.command_active["left"]
            ):
                self.command_active["forward"] = True
                keyboard.press(self.command_key_mappings["forward"])
            elif self.command_active["forward"]:
                self.command_active["forward"] = False
                keyboard.release(self.command_key_mappings["forward"])

    def process_duck_command(self, landmarks):
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        right_wrist_on_knee = self.is_point_on_knee(right_wrist, right_knee)
        left_wrist_on_knee = self.is_point_on_knee(left_wrist, left_knee)

        if right_wrist_on_knee and left_wrist_on_knee and not self.command_active["forward"]:
            self.command_active["duck"] = True
            keyboard.press(chr(self.command_key_mappings["duck"]))
        elif self.command_active["duck"]:
            self.command_active["duck"] = False
            keyboard.release(chr(self.command_key_mappings["duck"]))

    def is_point_on_knee(self, point, knee):
        if point is None or knee is None:
            return False

        threshold = 0.1  # Adjust the threshold value as per your requirements

        distance = self.calculate_distance(point, knee)
        return distance is not None and distance <= threshold

    def process_swing_command(self, landmarks):
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        if right_wrist is not None and left_wrist is not None and right_shoulder is not None and left_shoulder is not None:
            if (
                    self.are_points_touching(right_wrist, left_shoulder) and
                    self.are_points_touching(left_wrist, right_shoulder) and
                    not self.command_active["right"] and not self.command_active["left"] and
                    not self.command_active["forward"]
            ):
                self.command_active["swing"] = True
                mouse.mouseDown(button='left')
            elif self.command_active["swing"]:
                self.command_active["swing"] = False
                mouse.mouseUp(button='left')

    def are_points_touching(self, point1, point2, threshold=0.1):
        if point1 and point2:
            distance = math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)
            return distance <= threshold
        return False

    def are_points_overlapping(self, point1, point2, threshold=0.1):
        if point1 and point2:
            distance = math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)
            return distance <= threshold
        return False

    def is_body_leaning_forward(self, right_shoulder, left_shoulder, threshold=0.1):
        if right_shoulder and left_shoulder:
            return right_shoulder.y > left_shoulder.y
        return False

    def calculate_distance(self, a, b):
        if a and b:
            return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return None

    def send_key_event(self, key_code):
        keyboard.press(chr(key_code))


if __name__ == "__main__":
    window = Window()
    window.start_capture()
