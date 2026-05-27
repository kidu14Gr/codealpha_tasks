"""Real-time object detection and tracking with YOLO + SORT/DeepSORT."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from detector import YoloDetector
from tracking.deepsort_tracker import DeepSortTracker
from tracking.sort_tracker import SortTracker
from utils.visualization import FrameVisualizer


def parse_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    return source


def build_tracker(tracker_type: str):
    if tracker_type == "deepsort":
        return DeepSortTracker(max_age=config.MAX_AGE, min_hits=config.MIN_HITS), "DeepSORT"
    return SortTracker(
        max_age=config.MAX_AGE,
        min_hits=config.MIN_HITS,
        iou_threshold=config.SORT_IOU_THRESHOLD,
    ), "SORT"


def setup_trackbars(window: str, detector: YoloDetector) -> None:
    def on_conf(val):
        detector.set_confidence(val / 100.0)

    def on_iou(val):
        detector.set_iou(val / 100.0)

    cv2.createTrackbar("Confidence x100", window, int(config.CONFIDENCE_THRESHOLD * 100), 99, on_conf)
    cv2.createTrackbar("IOU x100", window, int(config.IOU_THRESHOLD * 100), 99, on_iou)


def run(
    source: str | int,
    tracker_type: str = config.TRACKER_TYPE,
    model_name: str = config.MODEL_NAME,
    record: bool = False,
    allowed_classes: list[str] | None = None,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    if allowed_classes:
        config.ALLOWED_CLASSES = allowed_classes

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    detector = YoloDetector(model_name=model_name)
    tracker, tracker_name = build_tracker(tracker_type)
    visualizer = FrameVisualizer()

    window = config.WINDOW_NAME
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    setup_trackbars(window, detector)

    writer = None
    recording = False
    paused = False
    frame_idx = 0
    log_path = config.LOG_DIR / f"tracks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        ["frame", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "timestamp"]
    )

    print("Controls: Q=quit, S=screenshot, R=toggle record, SPACE=pause")
    print(f"Logging detections to {log_path}")

    prev_time = time.time()
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if isinstance(source, int):
                        continue
                    print("End of video stream.")
                    break
                frame_idx += 1

                boxes, class_ids, class_names, confidences = detector.detect(frame)

                detections = np.zeros((0, 5))
                if len(boxes):
                    detections = np.hstack([boxes, confidences.reshape(-1, 1)])

                tracks = tracker.update(detections, class_ids, class_names, confidences)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                for trk in tracks:
                    x1, y1, x2, y2 = trk.bbox
                    csv_writer.writerow(
                        [
                            frame_idx,
                            trk.track_id,
                            trk.class_name,
                            f"{trk.confidence:.4f}",
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            datetime.now().isoformat(),
                        ]
                    )

                display = visualizer.draw(
                    frame, tracks, fps, paused=paused, recording=recording, tracker_name=tracker_name
                )
            else:
                display = visualizer.draw(
                    frame, [], 0.0, paused=True, recording=recording, tracker_name=tracker_name
                )

            if recording:
                if writer is None:
                    out_path = config.OUTPUT_DIR / f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, config.OUTPUT_FPS, (w, h))
                    print(f"Recording -> {out_path}")
                writer.write(display)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                shot = config.OUTPUT_DIR / f"{config.SCREENSHOT_PREFIX}_{frame_idx}.jpg"
                cv2.imwrite(str(shot), display)
                print(f"Screenshot saved: {shot}")
            if key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("Recording stopped.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        log_file.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Object Detection and Tracking")
    parser.add_argument(
        "--source",
        default=str(config.DEFAULT_SOURCE),
        help="Webcam index (0) or path to video file",
    )
    parser.add_argument(
        "--tracker",
        choices=["sort", "deepsort"],
        default=config.TRACKER_TYPE,
        help="Tracking algorithm",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_NAME,
        help="YOLO model weights (yolov8n.pt, yolov8s.pt, yolov8m.pt)",
    )
    parser.add_argument("--record", action="store_true", help="Start recording immediately")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Filter classes, e.g. --classes person car",
    )
    args = parser.parse_args()

    src = parse_source(args.source)
    run(
        source=src,
        tracker_type=args.tracker,
        model_name=args.model,
        record=args.record,
        allowed_classes=args.classes,
    )


if __name__ == "__main__":
    main()
