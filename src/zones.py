import cv2
import config


class ZoneAnalyzer:

    def __init__(self):
        print("[Zones] Zone analysis module ready.")

    def analyze(self, frame, centers, roi_box=None):
        h, w = frame.shape[:2]

        if roi_box is None:
            roi_box = (0, 0, w, h)

        x1, y1, x2, y2 = roi_box

        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        zone_w = max(1, roi_w // config.ZONE_COLS)
        zone_h = max(1, roi_h // config.ZONE_ROWS)

        zone_counts = {}
        crowded_zones = []

        for row in range(config.ZONE_ROWS):
            for col in range(config.ZONE_COLS):
                zone_counts[(row, col)] = 0

        for cx, cy in centers:
            if not (x1 <= cx < x2 and y1 <= cy < y2):
                continue

            local_x = cx - x1
            local_y = cy - y1

            col = min(local_x // zone_w, config.ZONE_COLS - 1)
            row = min(local_y // zone_h, config.ZONE_ROWS - 1)

            zone_counts[(row, col)] += 1

        for key, value in zone_counts.items():
            if value >= config.ZONE_CROWD_THRESHOLD:
                crowded_zones.append(key)

        return zone_counts, crowded_zones, roi_box

    def display(self, frame, zone_counts, crowded_zones, roi_box=None):
        h, w = frame.shape[:2]

        if roi_box is None:
            roi_box = (0, 0, w, h)

        x1, y1, x2, y2 = roi_box

        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        zone_w = max(1, roi_w // config.ZONE_COLS)
        zone_h = max(1, roi_h // config.ZONE_ROWS)

        for row in range(config.ZONE_ROWS):
            for col in range(config.ZONE_COLS):
                zx1 = x1 + col * zone_w
                zy1 = y1 + row * zone_h

                zx2 = x2 if col == config.ZONE_COLS - 1 else zx1 + zone_w
                zy2 = y2 if row == config.ZONE_ROWS - 1 else zy1 + zone_h

                count = zone_counts.get((row, col), 0)
                is_crowded = (row, col) in crowded_zones

                color = (0, 255, 0)

                if is_crowded:
                    color = (0, 0, 255)

                thickness = 2 if is_crowded else 1

                cv2.rectangle(
                    frame,
                    (zx1, zy1),
                    (zx2, zy2),
                    color,
                    thickness
                )

                cv2.putText(
                    frame,
                    f"Z{row},{col}: {count}",
                    (zx1 + 8, zy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                if is_crowded:
                    cv2.putText(
                        frame,
                        "CROWDED",
                        (zx1 + 8, zy1 + 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        if config.SHOW_ZONE_TOTAL:
            cv2.putText(
                frame,
                f"Crowded Zones: {len(crowded_zones)}",
                (15, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (0, 0, 255) if len(crowded_zones) > 0 else config.TEXT_COLOR,
                2
            )

        return frame