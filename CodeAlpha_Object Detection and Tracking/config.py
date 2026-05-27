"""Central configuration for object detection and tracking."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SAMPLES_DIR = BASE_DIR / "samples"
ASSETS_DIR = BASE_DIR / "assets"
LOG_DIR = OUTPUT_DIR / "logs"

# Video input
DEFAULT_SOURCE = 0  # webcam index
DEFAULT_VIDEO_PATH = SAMPLES_DIR / "demo.mp4"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection (YOLO)
MODEL_NAME = "yolov8n.pt"  # yolov8n.pt | yolov8s.pt | yolov8m.pt
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
DEVICE = "cpu"  # auto-detect in app if cuda available

# Tracking
TRACKER_TYPE = "sort"  # sort | deepsort
MAX_AGE = 30
MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Visualization
TRAIL_LENGTH = 20
BOX_THICKNESS = 2
FONT_SCALE = 0.55
SHOW_FPS = True
SHOW_TRACK_ID = True
SHOW_CLASS_NAME = True
SHOW_CONFIDENCE = True
SHOW_TRAILS = True
SHOW_HUD = True

# Recording
RECORD_OUTPUT = False
OUTPUT_FPS = 20.0
SCREENSHOT_PREFIX = "capture"

# Class filter (empty = all COCO classes)
ALLOWED_CLASSES: list[str] = []  # e.g. ["person", "car", "dog"]

# UI trackbars
WINDOW_NAME = "VisionTrack AI — Object Detection & Tracking"
