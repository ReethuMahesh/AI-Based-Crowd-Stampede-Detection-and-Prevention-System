import cv2
import sys
import os
import config

from src.detector import PersonDetector
from src.density import CrowdDensity
from src.optical_flow import OpticalFlow
from src.risk import RiskCalculator
from src.zones import ZoneAnalyzer
from src.heatmap import HeatmapGenerator
from src.alert import AlertSystem
from src.roi_manager import ROIManager
from src.report_generator import ReportGenerator

from src.tracker import SimpleTracker
from src.bottleneck import BottleneckDetector
from src.fall_detector import FallDetector


tracker = SimpleTracker()
bottleneck_detector = BottleneckDetector()
fall_detector = FallDetector()


def save_high_risk_frame(frame, frame_number):
    if not getattr(config, "SAVE_HIGH_RISK_FRAMES", False):
        return

    interval = getattr(config, "HIGH_RISK_SAVE_INTERVAL", 30)

    if frame_number % interval != 0:
        return

    os.makedirs(config.HIGH_RISK_DIR, exist_ok=True)

    path = os.path.join(
        config.HIGH_RISK_DIR,
        f"high_risk_frame_{frame_number}.jpg"
    )

    cv2.imwrite(path, frame)


def draw_clean_boxes(frame, boxes):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            config.BBOX_COLOR,
            config.BBOX_THICKNESS
        )


def draw_fall_info(frame, tracks, fall_ids):
    for track_id in fall_ids:
        if track_id not in tracks:
            continue

        data = tracks[track_id]
        x1, y1, x2, y2 = data["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "FALL?",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )


def draw_basic_stats(frame, count, roi_count, density_label, motion_value, motion_label):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (390, 145),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.45,
        frame,
        0.55,
        0,
        frame
    )

    density_color = (0, 255, 0)

    if density_label == "Medium":
        density_color = (0, 165, 255)
    elif density_label == "High":
        density_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"People: {count} | ROI: {roi_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.TEXT_COLOR,
        1
    )

    cv2.putText(
        frame,
        f"Density: {density_label}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        density_color,
        1
    )

    cv2.putText(
        frame,
        f"Motion: {motion_value:.2f} | {motion_label}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if motion_label == "Fast" else (0, 255, 0),
        1
    )


def main():
    print("=" * 60)
    print(" CROWD STAMPEDE EARLY WARNING SYSTEM ")
    print("=" * 60)

    if not os.path.exists(config.VIDEO_PATH):
        print(f"[ERROR] Video not found: {config.VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(config.VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = max(1, min(fps, 60))

    print(f"[Video] Resolution   : {frame_width} x {frame_height}")
    print(f"[Video] FPS          : {fps}")
    print(f"[Video] Total Frames : {total_frames}")
    print(f"[Mode] Normal YOLO every : {config.NORMAL_PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"[Mode] Danger YOLO every : {config.DANGER_PROCESS_EVERY_N_FRAMES} frame(s)")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    out = None

    if config.SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            config.OUTPUT_PATH,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

    detector = PersonDetector()
    density = CrowdDensity()
    flow = OpticalFlow()
    risk = RiskCalculator()
    zones = ZoneAnalyzer()
    heatmap = HeatmapGenerator(frame_width, frame_height)
    alert = AlertSystem()
    roi_mgr = ROIManager()

    report = ReportGenerator() if config.ENABLE_REPORT else None

    frame_number = 0
    processed_frames = 0

    show_heatmap = False
    paused = False

    frame = None
    display_frame = None

    last_count = 0
    last_boxes = []
    last_centers = []
    last_risk_level = config.RISK_SAFE

    max_people = 0
    max_risk_score = 0
    high_risk_frames = 0

    print("=" * 60)
    print("Controls:")
    print("Q = Quit")
    print("H = Heatmap ON/OFF")
    print("P = Pause")
    print("=" * 60)

    if config.SHOW_LIVE:
        cv2.namedWindow(
            "Crowd Analysis System",
            cv2.WINDOW_NORMAL
        )

    while True:

        if not paused:

            ret, frame = cap.read()

            if not ret:
                print("[Main] Video ended.")
                break

            original_frame = frame.copy()
            display_frame = original_frame.copy()

            frame_number += 1

            current_skip = config.NORMAL_PROCESS_EVERY_N_FRAMES

            if last_risk_level in [config.RISK_MEDIUM, config.RISK_HIGH]:
                current_skip = config.DANGER_PROCESS_EVERY_N_FRAMES

            run_yolo = (
                frame_number == 1 or
                frame_number % current_skip == 0
            )

            if run_yolo:
                processing_frame = frame.copy()

                processing_frame, count, boxes, centers = detector.detect(
                    processing_frame
                )

                last_count = count
                last_boxes = boxes
                last_centers = centers

                processed_frames += 1

            else:
                count = last_count
                boxes = last_boxes
                centers = last_centers

            tracks = tracker.update(boxes)
            fall_count, fall_ids = fall_detector.detect(tracks)

            roi_box = roi_mgr.get_roi(frame, centers)

            density_label, density_score, roi_count, density_roi_box = density.calculate(
                frame,
                count,
                centers,
                roi_box
            )

            avg_magnitude, motion_label, flow_data = flow.calculate(
                frame,
                roi_box
            )

            zone_counts, crowded_zones, zone_roi_box = zones.analyze(
                frame,
                centers,
                roi_box
            )

            bottleneck_flag, bottleneck_reason = bottleneck_detector.detect(
                crowded_zones,
                motion_label,
                density_label
            )

            risk_level, risk_score, surge_flag, reasons = risk.calculate(
                count=roi_count,
                density_label=density_label,
                motion_label=motion_label,
                crowded_zones=crowded_zones,
                motion_value=avg_magnitude,
                flow_data=flow_data,
                bottleneck_flag=bottleneck_flag,
                fall_count=fall_count
            )

            last_risk_level = risk_level

            max_people = max(max_people, count)
            max_risk_score = max(max_risk_score, risk_score)

            if risk_level == config.RISK_HIGH:
                high_risk_frames += 1
                save_high_risk_frame(original_frame, frame_number)

            if frame_number % config.PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[Frame {frame_number}/{total_frames}] "
                    f"YOLO={'Yes' if run_yolo else 'No'} | "
                    f"Skip={current_skip} | "
                    f"People={count} | "
                    f"ROI={roi_count} | "
                    f"Density={density_label} | "
                    f"Movement={motion_label} | "
                    f"Motion={avg_magnitude:.2f} | "
                    f"Chaos={flow_data.get('chaos', 0):.2f} | "
                    f"Accel={flow_data.get('acceleration', 0):.2f} | "
                    f"Falls={fall_count} | "
                    f"Bottleneck={'Yes' if bottleneck_flag else 'No'} | "
                    f"Risk={risk_level} | "
                    f"Score={risk_score}"
                )

                if getattr(config, "PRINT_RISK_REASONS_IN_LOGS", False):
                    print(f"Reasons: {reasons}")

            if report is not None:
                report.log(
                    frame_number=frame_number,
                    people_count=count,
                    roi_count=roi_count,
                    density_label=density_label,
                    density_score=density_score,
                    motion_value=avg_magnitude,
                    motion_label=motion_label,
                    crowded_zone_count=len(crowded_zones),
                    surge=surge_flag,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    reasons=reasons
                )

            draw_clean_boxes(display_frame, boxes)

            if fall_count > 0:
                draw_fall_info(display_frame, tracks, fall_ids)

            if show_heatmap:
                heatmap.update(centers)
                display_frame = heatmap.generate(display_frame)

            display_frame = zones.display(
                display_frame,
                zone_counts,
                crowded_zones,
                zone_roi_box
            )

            draw_basic_stats(
                display_frame,
                count,
                roi_count,
                density_label,
                avg_magnitude,
                motion_label
            )

            display_frame = risk.display(
                display_frame,
                risk_level,
                risk_score,
                surge_flag,
                len(crowded_zones),
                reasons
            )

            if alert.update(risk_level):
                display_frame = alert.display(display_frame)

            cv2.putText(
                display_frame,
                f"Frame: {frame_number}/{total_frames}",
                (frame_width - 260, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.TEXT_COLOR,
                1
            )

            if config.SAVE_OUTPUT and out is not None:
                out.write(display_frame)

        if config.SHOW_LIVE and display_frame is not None:
            cv2.imshow(
                "Crowd Analysis System",
                display_frame
            )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[Main] Quit pressed.")
            break

        elif key == ord("h"):
            show_heatmap = not show_heatmap
            print(f"[Main] Heatmap: {'ON' if show_heatmap else 'OFF'}")

        elif key == ord("p"):
            paused = not paused
            print("[Main] Paused" if paused else "[Main] Resumed")

    cap.release()

    if out:
        out.release()

    if report is not None:
        report.close()

    cv2.destroyAllWindows()

    print("=" * 60)
    print("CROWD ANALYSIS COMPLETED")
    print(f"Total frames read       : {frame_number}")
    print(f"YOLO processed frames   : {processed_frames}")
    print(f"Maximum people count    : {max_people}")
    print(f"Maximum risk score      : {max_risk_score}")
    print(f"High-risk frames        : {high_risk_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()