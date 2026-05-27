"""Parse MIDI files and build integer sequences for model training."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from music21 import chord, converter, note

import config


def _infer_genre_from_path(path: Path) -> int:
    name = path.stem.lower()
    for idx, genre in enumerate(config.GENRES):
        if genre in name:
            return idx
    return 0


def parse_midi_file(path: Path) -> list[str]:
    """Extract note/chord/rest symbols from a MIDI file."""
    parsed = converter.parse(str(path))
    symbols: list[str] = []

    for element in parsed.flat.notes:
        if isinstance(element, note.Note):
            symbols.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            symbols.append(".".join(sorted(element.pitches.pitchNames)))
        elif isinstance(element, note.Rest):
            symbols.append("REST")

    return symbols


def build_vocabulary(all_symbols: list[str]) -> dict[str, int]:
    unique = sorted(set(all_symbols))
    vocab = {symbol: idx + 1 for idx, symbol in enumerate(unique)}
    vocab["<PAD>"] = 0
    return vocab


def create_sequences(
    symbols: list[str], vocab: dict[str, int], seq_len: int, step: int
) -> tuple[list[list[int]], list[int]]:
    encoded = [vocab[s] for s in symbols if s in vocab]
    inputs: list[list[int]] = []
    targets: list[int] = []

    for i in range(0, len(encoded) - seq_len, step):
        chunk = encoded[i : i + seq_len]
        target = encoded[i + seq_len]
        if len(chunk) == seq_len:
            inputs.append(chunk)
            targets.append(target)

    return inputs, targets


def preprocess(save: bool = True) -> dict:
    midi_files = list(config.MIDI_DIR.rglob("*.mid")) + list(config.MIDI_DIR.rglob("*.midi"))
    if len(midi_files) < config.MIN_MIDI_FILES:
        raise FileNotFoundError(
            f"Need at least {config.MIN_MIDI_FILES} MIDI files in {config.MIDI_DIR}. "
            "Run: python download_data.py"
        )

    all_symbols: list[str] = []
    file_symbols: list[tuple[Path, list[str], int]] = []

    for path in midi_files:
        try:
            symbols = parse_midi_file(path)
            if len(symbols) < config.MIN_NOTES_PER_FILE:
                continue
            genre_id = _infer_genre_from_path(path)
            all_symbols.extend(symbols)
            file_symbols.append((path, symbols, genre_id))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not all_symbols:
        raise RuntimeError("No valid symbols extracted from MIDI files.")

    vocab = build_vocabulary(all_symbols)
    inv_vocab = {v: k for k, v in vocab.items()}

    all_inputs: list[list[int]] = []
    all_targets: list[int] = []
    all_genres: list[int] = []

    for _, symbols, genre_id in file_symbols:
        inputs, targets = create_sequences(
            symbols, vocab, config.SEQUENCE_LENGTH, config.STEP_SIZE
        )
        all_inputs.extend(inputs)
        all_targets.extend(targets)
        all_genres.extend([genre_id] * len(targets))

    payload = {
        "inputs": all_inputs,
        "targets": all_targets,
        "genres": all_genres,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "vocab_size": len(vocab),
        "num_genres": len(config.GENRES),
        "genre_names": list(config.GENRES),
    }

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with config.NOTES_PKL.open("wb") as f:
            pickle.dump(
                {
                    "inputs": all_inputs,
                    "targets": all_targets,
                    "genres": all_genres,
                },
                f,
            )
        with config.VOCAB_PKL.open("wb") as f:
            pickle.dump(
                {
                    "vocab": vocab,
                    "inv_vocab": inv_vocab,
                    "vocab_size": len(vocab),
                    "num_genres": len(config.GENRES),
                    "genre_names": list(config.GENRES),
                },
                f,
            )
        print(
            f"Saved {len(all_inputs)} training pairs | vocab size={len(vocab)} "
            f"to {config.NOTES_PKL}"
        )

    return payload


def load_preprocessed() -> dict:
    if not config.NOTES_PKL.exists() or not config.VOCAB_PKL.exists():
        return preprocess(save=True)

    with config.NOTES_PKL.open("rb") as f:
        notes = pickle.load(f)
    with config.VOCAB_PKL.open("rb") as f:
        vocab_data = pickle.load(f)

    return {**notes, **vocab_data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MIDI dataset")
    parser.add_argument("--force", action="store_true", help="Re-run preprocessing")
    args = parser.parse_args()

    if args.force or not config.NOTES_PKL.exists():
        preprocess(save=True)
    else:
        data = load_preprocessed()
        print(f"Using cached data: {len(data['inputs'])} pairs.")


if __name__ == "__main__":
    main()
