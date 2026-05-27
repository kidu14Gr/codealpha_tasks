"""YOLO-based object detector wrapper."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

import config


class YoloDetector:
    """Pre-trained YOLO detector via Ultralytics."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        conf: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(model_name)
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Run detection on a BGR frame.
        Returns: boxes (N,4), class_ids (N,), class_names, confidences (N,)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            empty = np.zeros((0, 4))
            return empty, np.array([]), [], np.array([])

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = [self.class_names[cid] for cid in class_ids]

        if config.ALLOWED_CLASSES:
            mask = np.array([n in config.ALLOWED_CLASSES for n in names])
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            names = [n for n, keep in zip(names, mask) if keep]

        return boxes, class_ids, names, confidences

    def set_confidence(self, conf: float) -> None:
        self.conf = max(0.05, min(conf, 0.99))

    def set_iou(self, iou: float) -> None:
        self.iou = max(0.1, min(iou, 0.99))
