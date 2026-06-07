import config


class BottleneckDetector:
    def __init__(self):
        print("[Bottleneck] Bottleneck detection ready.")
        self.history = []
        self.history_size = 10

    def detect(self, crowded_zones, motion_label, density_label):
        crowded_count = len(crowded_zones)

        bottleneck = False
        reason = "No bottleneck"

        if density_label == "High" and crowded_count >= 4 and motion_label == "Normal":
            bottleneck = True
            reason = "High density with low movement indicates crowd blockage"

        elif density_label == "High" and crowded_count >= 5:
            bottleneck = True
            reason = "Many zones are congested, possible bottleneck"

        self.history.append(bottleneck)

        if len(self.history) > self.history_size:
            self.history.pop(0)

        continuous_bottleneck = self.history.count(True) >= 5

        return continuous_bottleneck, reason