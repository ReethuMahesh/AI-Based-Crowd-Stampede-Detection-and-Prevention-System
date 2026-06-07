class FallDetector:

    def __init__(self):
        print("[FallDetector] Safer fall detection ready.")
        self.fall_history = {}

    def detect(self, tracks):
        fall_count = 0
        fall_ids = []

        for track_id, data in tracks.items():
            x1, y1, x2, y2 = data["box"]

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            aspect_ratio = width / height
            area = width * height

            # Safer fall logic:
            # A fallen person usually appears very wide and short.
            # This avoids false fall detection in top-view CCTV videos.
            possible_fall = (
                aspect_ratio > 2.2 and
                width > 60 and
                height < 45 and
                area > 1500
            )

            if possible_fall:
                self.fall_history[track_id] = self.fall_history.get(track_id, 0) + 1
            else:
                self.fall_history[track_id] = 0

            # Must look fallen continuously for many frames
            if self.fall_history[track_id] >= 8:
                fall_count += 1
                fall_ids.append(track_id)

        return fall_count, fall_ids