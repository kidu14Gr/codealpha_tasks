"""Create a demo GIF for the README from a recorded video or synthetic preview."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_GIF = ASSETS_DIR / "demo.gif"


def find_latest_recording() -> Path | None:
    videos = sorted(OUTPUT_DIR.glob("tracked_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def synthetic_frames(width: int = 720, height: int = 405, count: int = 36) -> list[np.ndarray]:
    """Build a lightweight preview when no recording exists yet."""
    frames: list[np.ndarray] = []
    for i in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (18, 22, 38)

        cv2.rectangle(frame, (12, 12), (340, 108), (28, 34, 58), -1)
        cv2.putText(
            frame,
            "VisionTrack AI",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Tracker: SORT | Objects: 2",
            (24, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 180, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {22 + (i % 5)}.4 | LIVE",
            (24, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 180, 210),
            1,
            cv2.LINE_AA,
        )

        x1 = 80 + int(4 * i)
        y1 = 140
        x2, y2 = x1 + 130, y1 + 170
        color = (88, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 24), (x1 + 170, y1), color, -1)
        cv2.putText(
            frame,
            "ID 1 | person | 0.91",
            (x1 + 6, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        trail = [(x1 + 65, y1 + 85 - t * 3) for t in range(8)]
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j - 1], trail[j], color, 2)

        x1b = 380 - int(3 * i)
        cv2.rectangle(frame, (x1b, 180), (x1b + 100, 300), (120, 220, 160), 2)
        cv2.putText(
            frame,
            "ID 2 | car | 0.87",
            (x1b + 4, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 255, 230),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Q quit | S screenshot | R record | SPACE pause",
            (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 195, 210),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    return frames


def video_to_frames(video_path: Path, max_frames: int = 48, target_width: int = 720) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(total // max_frames, 1)
    frames: list[np.ndarray] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            scale = target_width / w
            resized = cv2.resize(frame, (target_width, int(h * scale)))
            frames.append(resized)
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def save_gif(frames: list[np.ndarray], output_path: Path, fps: int = 10) -> None:
    try:
        import imageio.v3 as iio

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        iio.imwrite(output_path, rgb_frames, duration=1 / fps, loop=0)
        return
    except ImportError:
        pass

    try:
        from PIL import Image

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        return
    except ImportError as exc:
        raise RuntimeError(
            "Install imageio or Pillow to export GIFs: pip install imageio pillow"
        ) from exc


def synthetic_frames_pil(width: int = 720, height: int = 405, count: int = 36):
    """Pillow-only synthetic preview (no OpenCV required)."""
    from PIL import Image, ImageDraw

    frames = []
    for i in range(count):
        img = Image.new("RGB", (width, height), (18, 22, 38))
        d = ImageDraw.Draw(img)
        d.rectangle((12, 12, 340, 108), fill=(28, 34, 58))
        d.text((24, 20), "VisionTrack AI", fill=(235, 235, 245))
        d.text((24, 48), "Tracker: SORT | Objects: 2", fill=(170, 180, 210))
        d.text((24, 68), f"FPS: {22 + i % 5}.4 | LIVE", fill=(170, 180, 210))
        x1 = 80 + 4 * i
        d.rectangle((x1, 140, x1 + 130, 310), outline=(88, 140, 255), width=2)
        d.rectangle((x1, 116, x1 + 170, 140), fill=(88, 140, 255))
        d.text((x1 + 6, 118), "ID 1 | person | 0.91", fill=(255, 255, 255))
        x1b = 380 - 3 * i
        d.rectangle((x1b, 180, x1b + 100, 300), outline=(120, 220, 160), width=2)
        d.text((x1b + 4, 162), "ID 2 | car | 0.87", fill=(220, 255, 230))
        d.text((12, height - 22), "Q quit | S screenshot | R record | SPACE pause", fill=(190, 195, 210))
        frames.append(img)
    return frames


def save_gif_pil(pil_frames, output_path: Path, fps: int = 10) -> None:
    from PIL import Image

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create README demo GIF")
    parser.add_argument("--video", type=str, default=None, help="Input MP4 path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_GIF))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic preview")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    video_path = Path(args.video) if args.video else find_latest_recording()

    if args.synthetic or video_path is None or not HAS_CV2:
        if video_path and not args.synthetic and not HAS_CV2:
            print("OpenCV not installed; using synthetic preview.")
        else:
            print("Generating synthetic preview GIF...")
        save_gif_pil(synthetic_frames_pil(), output_path, fps=args.fps)
    else:
        print(f"Converting recording: {video_path}")
        frames = video_to_frames(video_path)
        save_gif(frames, output_path, fps=args.fps)

    print(f"Saved demo GIF -> {output_path}")


if __name__ == "__main__":
    main()
