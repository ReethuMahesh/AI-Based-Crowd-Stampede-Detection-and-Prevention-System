import cv2
import numpy as np
import config


class HeatmapGenerator:

    def __init__(self, frame_width, frame_height):
        print("[Heatmap] Heatmap module ready.")

        self.heat = np.zeros(
            (frame_height, frame_width),
            dtype=np.float32
        )

        self.frame_width = frame_width
        self.frame_height = frame_height

    def update(self, centers):
        self.heat *= 0.93

        for cx, cy in centers:
            if 0 <= cx < self.frame_width and 0 <= cy < self.frame_height:
                cv2.circle(
                    self.heat,
                    (cx, cy),
                    config.HEATMAP_RADIUS,
                    1.0,
                    -1
                )

    def generate(self, frame):
        heat_display = np.zeros_like(self.heat, dtype=np.uint8)

        if self.heat.max() > 0:
            cv2.normalize(
                self.heat,
                heat_display,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

        heatmap_coloured = cv2.applyColorMap(
            heat_display,
            cv2.COLORMAP_JET
        )

        mask = heat_display > 15
        mask_3ch = np.stack([mask, mask, mask], axis=2)

        blended = frame.copy()

        blended[mask_3ch] = cv2.addWeighted(
            heatmap_coloured,
            config.HEATMAP_ALPHA,
            frame,
            1 - config.HEATMAP_ALPHA,
            0
        )[mask_3ch]

        return blended