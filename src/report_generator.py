import os
import csv
import config


class ReportGenerator:

    def __init__(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        self.path = config.REPORT_PATH

        self.file = open(
            self.path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "frame_number",
            "people_count",
            "roi_count",
            "density_label",
            "density_score",
            "motion_value",
            "motion_label",
            "crowded_zone_count",
            "surge",
            "risk_score",
            "risk_level",
            "reasons"
        ])

        print(f"[Report] Saving CSV report to: {self.path}")

    def log(
        self,
        frame_number,
        people_count,
        roi_count,
        density_label,
        density_score,
        motion_value,
        motion_label,
        crowded_zone_count,
        surge,
        risk_score,
        risk_level,
        reasons
    ):

        self.writer.writerow([
            frame_number,
            people_count,
            roi_count,
            density_label,
            round(float(density_score), 4),
            round(float(motion_value), 4),
            motion_label,
            crowded_zone_count,
            "Yes" if surge else "No",
            risk_score,
            risk_level,
            " | ".join(reasons or [])
        ])

    def close(self):
        if not self.file.closed:
            self.file.close()
            print(f"[Report] Report saved: {self.path}")