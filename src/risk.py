import cv2
import config


class RiskCalculator:
    def __init__(self):
        print("[Risk] Fast early-warning risk module ready.")
        self.count_history = []
        self.motion_history = []
        self.history_size = 15

    def _add_reason(self, reasons, text):
        if text not in reasons:
            reasons.append(text)

    def calculate(
        self,
        count,
        density_label,
        motion_label,
        crowded_zones,
        motion_value=0.0,
        flow_data=None,
        bottleneck_flag=False,
        fall_count=0
    ):
        crowded_zone_count = len(crowded_zones)
        reasons = []

        if flow_data is None:
            flow_data = {}

        chaos = flow_data.get("chaos", 0.0)
        acceleration = flow_data.get("acceleration", 0.0)
        direction_consistency = flow_data.get("direction_consistency", 0.0)
        p90_speed = flow_data.get("p90_speed", 0.0)

        self.count_history.append(count)
        if len(self.count_history) > self.history_size:
            self.count_history.pop(0)

        self.motion_history.append(motion_value)
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)

        avg_count = sum(self.count_history) / max(1, len(self.count_history))

        surge_flag = False
        if len(self.count_history) >= 5:
            if count > avg_count * config.SURGE_COUNT_RATIO:
                surge_flag = True
                self._add_reason(reasons, "Sudden crowd increase detected")

        count_instability = 0.0
        if len(self.count_history) >= 6 and avg_count > 0:
            count_range = max(self.count_history) - min(self.count_history)
            count_instability = count_range / avg_count

        if density_label == "Low":
            density_score = 1
            self._add_reason(reasons, "Crowd density is low")
        elif density_label == "Medium":
            density_score = 3
            self._add_reason(reasons, "Moderate crowd detected")
        else:
            density_score = 5
            self._add_reason(reasons, "Large crowd detected")

        if motion_value >= config.MOTION_ABNORMAL_MIN or p90_speed >= config.MOTION_ABNORMAL_MIN:
            motion_score = 5
            dangerous_motion = True
            self._add_reason(reasons, "Abnormal fast movement detected")
        elif motion_label == "Fast" or motion_value >= config.MOTION_FAST_MIN:
            motion_score = 3
            dangerous_motion = True
            self._add_reason(reasons, "People are moving fast")
        else:
            motion_score = 1
            dangerous_motion = False
            self._add_reason(reasons, "Movement is normal")

        if crowded_zone_count >= 5:
            zone_score = 4
            self._add_reason(reasons, f"Many crowded zones detected: {crowded_zone_count}")
        elif crowded_zone_count >= 3:
            zone_score = 3
            self._add_reason(reasons, f"Some crowded zones detected: {crowded_zone_count}")
        elif crowded_zone_count >= 1:
            zone_score = 1
            self._add_reason(reasons, "Small crowded zone detected")
        else:
            zone_score = 0
            self._add_reason(reasons, "Crowd spread is normal")

        surge_score = 5 if surge_flag else 0

        acceleration_score = 0
        if acceleration >= config.ACCELERATION_THRESHOLD:
            acceleration_score = 5
            self._add_reason(reasons, "Sudden movement acceleration detected")

        chaos_score = 0
        if chaos >= config.CHAOS_MOTION_MIN:
            chaos_score = 5
            self._add_reason(reasons, "Chaotic crowd motion detected")

        direction_score = 0
        if direction_consistency >= config.DIRECTION_CONSISTENCY_THRESHOLD:
            direction_score = 4
            self._add_reason(reasons, "Crowd moving strongly in one direction")

        bottleneck_score = 0
        if bottleneck_flag and dangerous_motion:
            bottleneck_score = 5
            self._add_reason(reasons, "Bottleneck pressure with fast movement detected")

        fall_score = 0
        if fall_count > 0:
            fall_score = 4
            self._add_reason(reasons, f"Possible fallen person detected: {fall_count}")

        score = (
            density_score * 0.18 +
            motion_score * 0.22 +
            zone_score * 0.10 +
            surge_score * 0.14 +
            acceleration_score * 0.12 +
            chaos_score * 0.12 +
            direction_score * 0.05 +
            bottleneck_score * 0.05 +
            fall_score * 0.02
        ) * 2

        score = round(score, 2)

        # Strong early-warning override
        if (
            density_label in ["Medium", "High"]
            and (
                chaos >= config.CHAOS_MOTION_MIN
                or acceleration >= config.ACCELERATION_THRESHOLD
                or motion_value >= config.MOTION_ABNORMAL_MIN
                or p90_speed >= config.MOTION_ABNORMAL_MIN
            )
        ):
            return config.RISK_HIGH, max(score, 7.0), True, [
                "Early stampede warning: dense crowd with abnormal motion",
                "Chaotic or sudden movement detected",
                "Immediate attention required"
            ]

        if (
            density_label == "High"
            and dangerous_motion
            and crowded_zone_count >= 2
        ):
            return config.RISK_HIGH, max(score, 6.8), surge_flag, [
                "High density with fast movement",
                "Multiple crowded zones detected",
                "Possible stampede pressure building"
            ]

        if fall_count >= 2 and density_label in ["Medium", "High"]:
            return config.RISK_HIGH, max(score, 7.0), surge_flag, [
                "Multiple possible fallen persons detected",
                "Dense crowd around fall area",
                "High danger situation"
            ]

        # Normal dense crowd rule
        if (
            density_label == "High"
            and not dangerous_motion
            and count_instability < 0.20
            and fall_count == 0
        ):
            return config.RISK_DENSE, score, False, [
                "Large crowd detected",
                "Movement is stable",
                "Looks like dense but controlled crowd"
            ]

        if score >= 5.5 and dangerous_motion:
            risk_level = config.RISK_MEDIUM
        elif density_label == "Medium" and dangerous_motion:
            risk_level = config.RISK_MEDIUM
        else:
            risk_level = config.RISK_SAFE

        if risk_level == config.RISK_SAFE:
            reasons = ["Risk is safe because crowd behaviour is stable"]

        return risk_level, score, surge_flag, reasons

    def get_color(self, risk_level):
        if risk_level == config.RISK_SAFE:
            return (0, 255, 0)
        if risk_level == config.RISK_DENSE:
            return (255, 255, 0)
        if risk_level == config.RISK_MEDIUM:
            return (0, 165, 255)
        return (0, 0, 255)

    def display(
        self,
        frame,
        risk_level,
        risk_score,
        surge_flag,
        crowded_zone_count,
        reasons=None
    ):
        h, w = frame.shape[:2]
        color = self.get_color(risk_level)

        panel_x1 = 10
        panel_y1 = h - 185
        panel_x2 = 560
        panel_y2 = h - 10

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y = panel_y1 + 25

        cv2.putText(
            frame,
            f"Risk: {risk_level} | Score: {risk_score} | Surge: {'Yes' if surge_flag else 'No'}",
            (panel_x1 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        y += 27

        cv2.putText(
            frame,
            f"Crowded Zones: {crowded_zone_count}",
            (panel_x1 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            config.TEXT_COLOR,
            1
        )

        y += 25

        cv2.putText(
            frame,
            "Why:",
            (panel_x1 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 0),
            1
        )

        y += 22

        if reasons:
            for reason in reasons[:3]:
                short_reason = reason[:58] + "..." if len(reason) > 58 else reason

                cv2.putText(
                    frame,
                    f"- {short_reason}",
                    (panel_x1 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

                y += 20

        if risk_level == config.RISK_HIGH:
            cv2.rectangle(
                frame,
                (0, 0),
                (w - 1, h - 1),
                (0, 0, 255),
                5
            )

        return frame