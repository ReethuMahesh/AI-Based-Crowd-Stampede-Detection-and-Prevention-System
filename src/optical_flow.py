import cv2
import numpy as np
import config


class OpticalFlow:
    def __init__(self):
        print("[OpticalFlow] Fast stampede-motion module ready.")
        self.prev_gray = None
        self.motion_history = []
        self.history_size = 6

    def _empty_flow_data(self):
        return {
            "avg_speed": 0.0,
            "p90_speed": 0.0,
            "max_speed": 0.0,
            "chaos": 0.0,
            "acceleration": 0.0,
            "direction_consistency": 0.0
        }

    def _get_roi_gray(self, frame, roi_box=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if roi_box is None:
            roi_box = (0, 0, w, h)

        x1, y1, x2, y2 = roi_box

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))

        roi_gray = gray[y1:y2, x1:x2]

        # Smaller scale = faster optical flow
        roi_gray = cv2.resize(roi_gray, (0, 0), fx=0.35, fy=0.35)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        return roi_gray

    def calculate(self, frame, roi_box=None):
        curr_gray = self._get_roi_gray(frame, roi_box)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return 0.0, "Normal", self._empty_flow_data()

        if self.prev_gray.shape != curr_gray.shape:
            self.prev_gray = curr_gray
            return 0.0, "Normal", self._empty_flow_data()

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            curr_gray,
            None,
            0.5,
            2,
            13,
            2,
            5,
            1.1,
            0
        )

        fx = flow[:, :, 0]
        fy = flow[:, :, 1]

        magnitude, angle = cv2.cartToPolar(fx, fy)

        valid_motion = magnitude[magnitude > 0.12]

        if valid_motion.size == 0:
            avg_speed = 0.0
            p90_speed = 0.0
            max_speed = 0.0
            chaos = 0.0
            direction_consistency = 0.0
        else:
            avg_speed = float(np.mean(valid_motion))
            p90_speed = float(np.percentile(valid_motion, 90))
            max_speed = float(np.max(valid_motion))
            chaos = float(np.std(valid_motion))

            strong_mask = magnitude > np.percentile(valid_motion, 60)

            if np.sum(strong_mask) > 10:
                strong_angles = angle[strong_mask]

                mean_x = np.mean(np.cos(strong_angles))
                mean_y = np.mean(np.sin(strong_angles))

                direction_consistency = float(
                    np.sqrt(mean_x ** 2 + mean_y ** 2)
                )
            else:
                direction_consistency = 0.0

        self.motion_history.append(avg_speed)

        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)

        smoothed_speed = float(np.mean(self.motion_history))

        if len(self.motion_history) >= 2:
            acceleration = abs(self.motion_history[-1] - self.motion_history[-2])
        else:
            acceleration = 0.0

        motion_label = "Normal"

        if (
            smoothed_speed >= config.MOTION_FAST_MIN
            or p90_speed >= config.MOTION_FAST_MIN * 1.6
            or chaos >= config.CHAOS_MOTION_MIN
            or acceleration >= config.ACCELERATION_THRESHOLD
        ):
            motion_label = "Fast"

        flow_data = {
            "avg_speed": smoothed_speed,
            "p90_speed": p90_speed,
            "max_speed": max_speed,
            "chaos": chaos,
            "acceleration": acceleration,
            "direction_consistency": direction_consistency
        }

        self.prev_gray = curr_gray

        return smoothed_speed, motion_label, flow_data

    def display(self, frame, avg_magnitude, motion_label):
        color = (0, 255, 0) if motion_label == "Normal" else (0, 0, 255)

        cv2.putText(
            frame,
            f"Motion: {avg_magnitude:.2f}",
            (15, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Movement: {motion_label}",
            (15, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            color,
            2
        )

        return frame