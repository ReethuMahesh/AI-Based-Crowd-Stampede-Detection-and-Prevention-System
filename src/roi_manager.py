class ROIManager:

    def __init__(self):
        print("[ROI] Dynamic ROI manager ready.")

        self.prev_roi = None

        self.alpha = 0.7
        self.min_centers_for_roi = 5

        self.margin_x = 80
        self.margin_y = 80

    def _clamp(self, value, low, high):
        return max(low, min(value, high))

    def _smooth_roi(self, current_roi):
        if self.prev_roi is None:
            return current_roi

        px1, py1, px2, py2 = self.prev_roi
        cx1, cy1, cx2, cy2 = current_roi

        x1 = int(self.alpha * px1 + (1 - self.alpha) * cx1)
        y1 = int(self.alpha * py1 + (1 - self.alpha) * cy1)
        x2 = int(self.alpha * px2 + (1 - self.alpha) * cx2)
        y2 = int(self.alpha * py2 + (1 - self.alpha) * cy2)

        return x1, y1, x2, y2

    def get_roi(self, frame, centers):
        h, w = frame.shape[:2]

        if centers is None or len(centers) < self.min_centers_for_roi:
            if self.prev_roi is not None:
                return self.prev_roi

            return 0, 0, w, h

        xs = [cx for cx, cy in centers]
        ys = [cy for cx, cy in centers]

        min_x = self._clamp(min(xs) - self.margin_x, 0, w)
        max_x = self._clamp(max(xs) + self.margin_x, 0, w)

        min_y = self._clamp(min(ys) - self.margin_y, 0, h)
        max_y = self._clamp(max(ys) + self.margin_y, 0, h)

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w >= 0.80 * w or roi_h >= 0.80 * h:
            current_roi = (0, 0, w, h)
        else:
            current_roi = (min_x, min_y, max_x, max_y)

        smoothed_roi = self._smooth_roi(current_roi)

        x1, y1, x2, y2 = smoothed_roi

        x1 = self._clamp(x1, 0, w - 1)
        y1 = self._clamp(y1, 0, h - 1)
        x2 = self._clamp(x2, x1 + 1, w)
        y2 = self._clamp(y2, y1 + 1, h)

        final_roi = x1, y1, x2, y2

        self.prev_roi = final_roi

        return final_roi