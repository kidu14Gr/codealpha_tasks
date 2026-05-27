"""Download or synthesize MIDI training data for the music generation model."""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from music21 import chord, metadata, note, stream

import config

# Small public-domain style sample URLs (classical piano excerpts) — optional fetch
SAMPLE_URLS = [
    "https://www.midiworld.com/download/4505",  # may fail; fallback to synthesis
]


def _scale_notes(key_name: str) -> list[str]:
    scales = {
        "C": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
        "G": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4"],
        "F": ["F3", "G3", "A3", "Bb3", "C4", "D4", "E4", "F4"],
        "D": ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"],
    }
    return scales.get(key_name, scales["C"])


def _jazz_progression() -> list[str]:
    return ["C4", "E4", "G4", "B4", "D5", "F5", "A5", "C6"]


def _folk_melody() -> list[str]:
    return ["E4", "G4", "A4", "B4", "A4", "G4", "E4", "D4", "E4", "G4"]


def synthesize_midi(path: Path, genre: str, index: int) -> None:
    """Create a short coherent MIDI excerpt for training."""
    s = stream.Stream()
    s.insert(0, metadata.Metadata())
    s.metadata.title = f"Synthetic {genre} {index}"

    rng = random.Random(index + hash(genre) % 10000)
    key = rng.choice(["C", "G", "F", "D"])

    if genre == "classical":
        pitches = _scale_notes(key)
        for _ in range(rng.randint(40, 90)):
            p = rng.choice(pitches)
            n = note.Note(p, quarterLength=rng.choice([0.5, 0.75, 1.0, 1.0, 1.5]))
            n.volume.velocity = rng.randint(55, 95)
            s.append(n)
            if rng.random() < 0.12:
                c = chord.Chord(rng.sample(pitches, k=3), quarterLength=1.0)
                s.append(c)
    elif genre == "jazz":
        pitches = _jazz_progression()
        for _ in range(rng.randint(35, 80)):
            p = rng.choice(pitches)
            n = note.Note(p, quarterLength=rng.choice([0.5, 1.0, 1.5, 2.0]))
            n.volume.velocity = rng.randint(60, 100)
            s.append(n)
    else:
        pitches = _folk_melody()
        for _ in range(rng.randint(35, 75)):
            p = rng.choice(pitches)
            n = note.Note(p, quarterLength=rng.choice([0.5, 1.0, 1.0, 2.0]))
            n.volume.velocity = rng.randint(50, 90)
            s.append(n)

    if rng.random() < 0.15:
        s.append(note.Rest(quarterLength=0.5))

    s.write("midi", fp=str(path))


def try_download_maestro_subset(target_dir: Path, max_files: int = 20) -> int:
    """
    Attempt to download a small MAESTRO subset zip.
    Returns number of files successfully placed.
    """
    maestro_url = (
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
        "maestro-v3.0.0-midi.zip"
    )
    zip_path = target_dir.parent / "maestro_subset.zip"
    try:
        print("Attempting MAESTRO download (this may take a while)...")
        urlretrieve(maestro_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            midi_names = [n for n in zf.namelist() if n.lower().endswith(".mid")][:max_files]
            for name in midi_names:
                zf.extract(name, target_dir)
        zip_path.unlink(missing_ok=True)
        return len(list(target_dir.rglob("*.mid")))
    except Exception as exc:
        print(f"MAESTRO download skipped: {exc}")
        zip_path.unlink(missing_ok=True)
        return 0


def ensure_dataset(count: int | None = None) -> int:
    """Ensure at least MIN_MIDI_FILES exist; synthesize if needed."""
    target_dir = config.MIDI_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = list(target_dir.rglob("*.mid")) + list(target_dir.rglob("*.midi"))
    if len(existing) < config.MIN_MIDI_FILES:
        downloaded = try_download_maestro_subset(target_dir, max_files=25)
        existing = list(target_dir.rglob("*.mid")) + list(target_dir.rglob("*.midi"))
        print(f"After download attempt: {len(existing)} MIDI files (downloaded {downloaded}).")

    goal = count or config.TARGET_MIDI_FILES
    idx = len(existing)
    genres = list(config.GENRES)

    while len(existing) < max(goal, config.MIN_MIDI_FILES):
        genre = genres[idx % len(genres)]
        out = target_dir / f"{genre}_{idx:03d}.mid"
        synthesize_midi(out, genre, idx)
        existing.append(out)
        idx += 1

    print(f"Dataset ready: {len(existing)} MIDI files in {target_dir}")
    return len(existing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or synthesize MIDI dataset")
    parser.add_argument("--count", type=int, default=config.TARGET_MIDI_FILES)
    args = parser.parse_args()
    ensure_dataset(args.count)


if __name__ == "__main__":
    main()
