"""Drawing utilities for detection and tracking overlays."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from tracking.sort_tracker import TrackResult


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 17)
    return tuple(int(x) for x in rng.integers(60, 255, size=3))


class FrameVisualizer:
    """Render bounding boxes, labels, trails, and HUD."""

    def __init__(self):
        self.fps_history: List[float] = []
        self.trails: Dict[int, List[Tuple[int, int]]] = {}

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        fps: float,
        paused: bool = False,
        recording: bool = False,
        tracker_name: str = "SORT",
    ) -> np.ndarray:
        canvas = frame.copy()

        for trk in tracks:
            color = _color_for_id(trk.track_id)
            x1, y1, x2, y2 = map(int, trk.bbox)

            if config.SHOW_TRAILS:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trail = self.trails.setdefault(trk.track_id, [])
                trail.append((cx, cy))
                self.trails[trk.track_id] = trail[-config.TRAIL_LENGTH :]
                for i in range(1, len(self.trails[trk.track_id])):
                    cv2.line(
                        canvas,
                        self.trails[trk.track_id][i - 1],
                        self.trails[trk.track_id][i],
                        color,
                        2,
                    )

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

            label_parts = []
            if config.SHOW_TRACK_ID:
                label_parts.append(f"ID {trk.track_id}")
            if config.SHOW_CLASS_NAME:
                label_parts.append(trk.class_name)
            if config.SHOW_CONFIDENCE:
                label_parts.append(f"{trk.confidence:.2f}")

            label = " | ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if config.SHOW_HUD:
            self._draw_hud(canvas, tracks, fps, paused, recording, tracker_name)

        return canvas

    def _draw_hud(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        fps: float,
        paused: bool,
        recording: bool,
        tracker_name: str,
    ) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (min(w - 10, 360), 120), (20, 20, 30), -1)
        frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        lines = [
            "VisionTrack AI",
            f"Tracker: {tracker_name} | Objects: {len(tracks)}",
            f"FPS: {fps:.1f}" if config.SHOW_FPS else "",
            "REC" if recording else ("PAUSED" if paused else "LIVE"),
        ]
        y = 32
        for line in lines:
            if not line:
                continue
            color = (0, 0, 255) if line == "REC" else (240, 240, 240)
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 24

        help_text = "Q quit | S screenshot | R record | SPACE pause"
        cv2.putText(
            frame,
            help_text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
