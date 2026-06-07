import math


class SimpleTracker:
    def __init__(self, max_distance=60, max_missing=10):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, boxes):
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((box, cx, cy))

        assigned_tracks = set()
        assigned_detections = set()

        for track_id, track in list(self.tracks.items()):
            best_det = None
            best_dist = float("inf")

            tx, ty = track["center"]

            for i, (box, cx, cy) in enumerate(detections):
                if i in assigned_detections:
                    continue

                dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_det = i

            if best_det is not None and best_dist <= self.max_distance:
                box, cx, cy = detections[best_det]

                self.tracks[track_id]["box"] = box
                self.tracks[track_id]["center"] = (cx, cy)
                self.tracks[track_id]["missing"] = 0

                assigned_tracks.add(track_id)
                assigned_detections.add(best_det)

            else:
                self.tracks[track_id]["missing"] += 1

        for i, (box, cx, cy) in enumerate(detections):
            if i not in assigned_detections:
                self.tracks[self.next_id] = {
                    "box": box,
                    "center": (cx, cy),
                    "missing": 0
                }

                self.next_id += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["missing"] > self.max_missing:
                del self.tracks[track_id]

        return self.tracks