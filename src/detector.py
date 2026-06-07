import os
import cv2
import torch
import config
from ultralytics import YOLO


class PersonDetector:

    def __init__(self):
        print("[Detector] Loading YOLO model...")

        os.makedirs(config.MODELS_DIR, exist_ok=True)

        if os.path.exists(config.YOLO_MODEL_PATH):
            self.model = YOLO(config.YOLO_MODEL_PATH)
        else:
            self.model = YOLO(config.YOLO_MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model.to(self.device)
        except Exception:
            pass

        print(f"[Detector] Device : {self.device.upper()}")
        print(f"[Detector] Model  : {config.YOLO_MODEL_NAME}")
        print(f"[Detector] Tiling : {config.ENABLE_TILED_DETECTION}")

    def _run_yolo(self, image, conf, imgsz):
        results = self.model(
            image,
            conf=conf,
            iou=0.35,
            classes=[config.PERSON_CLASS_ID],
            imgsz=imgsz,
            device=self.device,
            agnostic_nms=True,
            max_det=getattr(config, "MAX_DETECTIONS", 5000),
            half=True if self.device == "cuda" else False,
            verbose=False
        )

        return results[0].boxes

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        union = area_a + area_b - inter_area

        return inter_area / max(1, union)

    def _nms(self, detections):
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda item: item[4],
            reverse=True
        )

        final = []
        threshold = config.DUPLICATE_IOU_THRESHOLD

        while detections:
            best = detections.pop(0)
            final.append(best)

            detections = [
                d for d in detections
                if self._iou(best[:4], d[:4]) < threshold
            ]

        return final

    def _valid_box(self, x1, y1, x2, y2, frame_w, frame_h):
        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w - 1))
        y2 = max(0, min(int(y2), frame_h - 1))

        bw = x2 - x1
        bh = y2 - y1

        if bw < config.MIN_PERSON_BOX_WIDTH:
            return None

        if bh < config.MIN_PERSON_BOX_HEIGHT:
            return None

        return x1, y1, x2, y2

    def _collect_boxes(self, image, offset_x, offset_y, frame_w, frame_h, conf, imgsz):
        collected = []

        boxes = self._run_yolo(
            image=image,
            conf=conf,
            imgsz=imgsz
        )

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            score = float(box.conf[0]) if box.conf is not None else 0.0

            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            valid = self._valid_box(
                x1,
                y1,
                x2,
                y2,
                frame_w,
                frame_h
            )

            if valid is None:
                continue

            vx1, vy1, vx2, vy2 = valid
            collected.append((vx1, vy1, vx2, vy2, score))

        return collected

    def _tile_windows(self, width, height):
        rows = config.TILE_ROWS
        cols = config.TILE_COLS
        overlap = config.TILE_OVERLAP

        tile_w = int(width / cols)
        tile_h = int(height / rows)

        step_x = max(1, int(tile_w * (1 - overlap)))
        step_y = max(1, int(tile_h * (1 - overlap)))

        windows = []

        y = 0

        while y < height:
            x = 0

            y2 = min(height, y + tile_h)
            y1 = max(0, y2 - tile_h)

            while x < width:
                x2 = min(width, x + tile_w)
                x1 = max(0, x2 - tile_w)

                windows.append((x1, y1, x2, y2))

                if x2 >= width:
                    break

                x += step_x

            if y2 >= height:
                break

            y += step_y

        return windows

    def detect(self, frame):
        height, width = frame.shape[:2]

        detections = []

        # Full-frame detection
        detections.extend(
            self._collect_boxes(
                image=frame,
                offset_x=0,
                offset_y=0,
                frame_w=width,
                frame_h=height,
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.YOLO_IMAGE_SIZE
            )
        )

        # Tiled detection for far/small CCTV people
        if config.ENABLE_TILED_DETECTION:
            for x1, y1, x2, y2 in self._tile_windows(width, height):
                tile = frame[y1:y2, x1:x2]

                if tile.size == 0:
                    continue

                detections.extend(
                    self._collect_boxes(
                        image=tile,
                        offset_x=x1,
                        offset_y=y1,
                        frame_w=width,
                        frame_h=height,
                        conf=config.TILE_CONFIDENCE,
                        imgsz=config.TILE_IMAGE_SIZE
                    )
                )

        final_detections = self._nms(detections)

        all_boxes = []
        all_centers = []

        for x1, y1, x2, y2, score in final_detections:
            all_boxes.append([x1, y1, x2, y2])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            all_centers.append((cx, cy))

        return frame, len(all_boxes), all_boxes, all_centers