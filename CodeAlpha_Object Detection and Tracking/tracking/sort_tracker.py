"""SORT: Simple Online and Realtime Tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection
    return intersection / (union + 1e-6)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert state [cx, cy, s, r] to [x1,y1,x2,y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2]).reshape((1, 4))


class KalmanBoxTracker:
    """Kalman filter based bounding box tracker for a single object."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: list[tuple[int, int]] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


@dataclass
class TrackResult:
    track_id: int
    bbox: np.ndarray  # x1,y1,x2,y2
    class_id: int
    class_name: str
    confidence: float


class SortTracker:
    """SORT tracker with class-aware association."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._class_map: dict[int, tuple[int, str, float]] = {}

    def update(
        self,
        detections: np.ndarray,
        class_ids: np.ndarray,
        class_names: list[str],
        confidences: np.ndarray,
    ) -> list[TrackResult]:
        """
        Update tracker with detections shaped (N, 5) as x1,y1,x2,y2,score.
        """
        self.frame_count += 1
        results: list[TrackResult] = []

        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = trks[: len(self.trackers)]

        matched, unmatched_dets, unmatched_trks = self._associate(detections, trks)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx, :4])
            cid = int(class_ids[det_idx])
            self._class_map[trk_idx] = (cid, class_names[det_idx], float(confidences[det_idx]))

        # Create new trackers
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))
            trk_idx = len(self.trackers) - 1
            cid = int(class_ids[i])
            self._class_map[trk_idx] = (cid, class_names[i], float(confidences[i]))

        # Build output
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                cid, cname, conf = self._class_map.get(i, (0, "object", 0.0))
                cx = int((d[0] + d[2]) / 2)
                cy = int((d[1] + d[3]) / 2)
                trk.history.append((cx, cy))
                results.append(
                    TrackResult(
                        track_id=trk.id + 1,
                        bbox=d,
                        class_id=cid,
                        class_name=cname,
                        confidence=conf,
                    )
                )
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                self._class_map.pop(i, None)

        return results

    def _associate(
        self, detections: np.ndarray, trackers: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched = []
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] < self.iou_threshold:
                        continue
                    matched.append((r, c))
                matched = matched
        else:
            matched = []

        matched_indices = list(matched)
        unmatched_dets = [d for d in range(len(detections)) if d not in [m[0] for m in matched_indices]]
        unmatched_trks = [t for t in range(len(trackers)) if t not in [m[1] for m in matched_indices]]
        return matched_indices, unmatched_dets, unmatched_trks

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        for trk in self.trackers:
            if trk.id + 1 == track_id:
                return trk.history[-20:]
        return []
