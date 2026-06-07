import cv2
import config


class CrowdDensity:

    def __init__(self):
        print("[Density] ROI density module ready.")

    def calculate(self, frame, count, centers=None, roi_box=None):
        h, w = frame.shape[:2]

        if roi_box is None:
            roi_box = (0, 0, w, h)

        x1, y1, x2, y2 = roi_box

        roi_width = max(1, x2 - x1)
        roi_height = max(1, y2 - y1)
        roi_area = roi_width * roi_height

        roi_count = count

        if centers is not None:
            roi_count = 0

            for cx, cy in centers:
                if x1 <= cx < x2 and y1 <= cy < y2:
                    roi_count += 1

        density_score = (roi_count / roi_area) * 100000

        if roi_count <= config.DENSITY_LOW_MAX:
            density_label = "Low"

        elif roi_count <= config.DENSITY_MEDIUM_MAX:
            density_label = "Medium"

        else:
            density_label = "High"

        return density_label, density_score, roi_count, roi_box

    def display(self, frame, count, density_label, density_score, roi_count=None, roi_box=None):
        if roi_count is None:
            roi_count = count

        cv2.putText(
            frame,
            f"People Count : {count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"ROI Count    : {roi_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 0),
            2
        )

        color = (0, 255, 0)

        if density_label == "Medium":
            color = (0, 165, 255)

        elif density_label == "High":
            color = (0, 0, 255)

        cv2.putText(
            frame,
            f"Density      : {density_label}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            color,
            2
        )

        cv2.putText(
            frame,
            f"DensityScore : {density_score:.2f}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            2
        )

        if roi_box is not None and config.DRAW_ROI_BOUNDARY:
            x1, y1, x2, y2 = roi_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2 - 1, y2 - 1),
                config.ROI_BOUNDARY_COLOR,
                2
            )

            cv2.putText(
                frame,
                "Dynamic Crowd ROI",
                (x1 + 10, max(25, y1 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                config.ROI_BOUNDARY_COLOR,
                2
            )

        return frame