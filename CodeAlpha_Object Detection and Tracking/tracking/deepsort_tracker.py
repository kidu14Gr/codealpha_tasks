"""DeepSORT wrapper with graceful fallback to SORT."""

from __future__ import annotations

from typing import List

import numpy as np

from tracking.sort_tracker import SortTracker, TrackResult


class DeepSortTracker:
    """DeepSORT tracker using deep_sort_realtime when installed."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self._fallback = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=0.3)
        self._engine = None
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._engine = DeepSort(max_age=max_age, n_init=min_hits)
            self._mode = "deepsort"
        except Exception:
            self._mode = "sort_fallback"

    def update(
        self,
        detections: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        confidences: np.ndarray,
    ) -> List[TrackResult]:
        if self._engine is None:
            return self._fallback.update(detections, class_ids, class_names, confidences)

        raw = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections[i, :4]
            w, h = x2 - x1, y2 - y1
            raw.append(([x1, y1, w, h], float(confidences[i]), class_names[i]))

        tracks = self._engine.update_tracks(raw, frame=None)
        results: List[TrackResult] = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            l, t, r, b = trk.to_ltrb()
            results.append(
                TrackResult(
                    track_id=int(trk.track_id),
                    bbox=np.array([l, t, r, b]),
                    class_id=0,
                    class_name=str(trk.det_class or "object"),
                    confidence=float(trk.det_conf or 0.0),
                )
            )
        return results

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        return self._fallback.get_trail(track_id)
