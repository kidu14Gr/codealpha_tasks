"""Generate new MIDI sequences from a trained model."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from music21 import chord, metadata, note, stream

import config
from models.architectures import build_model
from preprocess import load_preprocessed


def load_trained_model(device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {config.BEST_MODEL_PATH}. Run train.py first."
        )

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    data = load_preprocessed()
    model_type = checkpoint.get("model_type", config.MODEL_TYPE)
    model = build_model(model_type, data["vocab_size"], data["num_genres"], config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, data, device


def sample_next(
    logits: torch.Tensor, temperature: float, rng: random.Random
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
    probs = probs / probs.sum()
    choices = np.arange(len(probs))
    return int(rng.choices(choices, weights=probs, k=1)[0])


def symbols_to_stream(symbols: list[str], humanize: bool = True) -> stream.Stream:
    """Convert symbol list back to a music21 Stream."""
    s = stream.Stream()
    s.insert(0, metadata.Metadata())
    s.metadata.title = "AI Generated Sequence"
    rng = random.Random()

    for sym in symbols:
        if sym == "REST":
            s.append(note.Rest(quarterLength=0.5 + rng.uniform(0, 0.2)))
            continue
        if "." in sym:
            pitches = sym.split(".")
            c = chord.Chord(pitches, quarterLength=1.0)
            if humanize:
                c.volume.velocity = 70 + rng.randint(
                    -config.HUMANIZE_VELOCITY_RANGE, config.HUMANIZE_VELOCITY_RANGE
                )
                c.quarterLength += rng.uniform(
                    -config.HUMANIZE_DURATION_RANGE, config.HUMANIZE_DURATION_RANGE
                )
            s.append(c)
        else:
            n = note.Note(sym, quarterLength=1.0)
            if humanize:
                n.volume.velocity = 72 + rng.randint(
                    -config.HUMANIZE_VELOCITY_RANGE, config.HUMANIZE_VELOCITY_RANGE
                )
                n.quarterLength += rng.uniform(
                    -config.HUMANIZE_DURATION_RANGE, config.HUMANIZE_DURATION_RANGE
                )
            s.append(n)
    return s


def generate_sequence(
    num_notes: int = config.DEFAULT_NUM_NOTES,
    temperature: float = config.DEFAULT_TEMPERATURE,
    genre: str | None = None,
    seed_symbols: list[str] | None = None,
    humanize: bool = True,
) -> list[str]:
    model, data, device = load_trained_model()
    inv_vocab = data["inv_vocab"]
    vocab = data["vocab"]
    rng = random.Random()

    genre_id = 0
    if genre:
        genre_names = data.get("genre_names", list(config.GENRES))
        if genre.lower() in genre_names:
            genre_id = genre_names.index(genre.lower())

    if seed_symbols:
        seed_encoded = [vocab[s] for s in seed_symbols if s in vocab][-config.SEQUENCE_LENGTH :]
    else:
        seed_encoded = random.choice(data["inputs"])[: config.SEQUENCE_LENGTH]

    while len(seed_encoded) < config.SEQUENCE_LENGTH:
        seed_encoded.insert(0, 0)

    generated = list(seed_encoded)

    with torch.no_grad():
        for _ in range(num_notes):
            x = torch.tensor([generated[-config.SEQUENCE_LENGTH :]], dtype=torch.long).to(device)
            g = torch.tensor([genre_id], dtype=torch.long).to(device)
            logits = model(x, g)[0]
            next_id = sample_next(logits, temperature, rng)
            generated.append(next_id)

    output_ids = generated[config.SEQUENCE_LENGTH :]
    symbols = [inv_vocab[i] for i in output_ids if i in inv_vocab and inv_vocab[i] != "<PAD>"]
    return symbols


def save_midi(symbols: list[str], output_path: Path, humanize: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    music_stream = symbols_to_stream(symbols, humanize=humanize)
    music_stream.write("midi", fp=str(output_path))
    return output_path


def generate_batch(
    num_outputs: int = config.DEFAULT_NUM_OUTPUTS,
    num_notes: int = config.DEFAULT_NUM_NOTES,
    temperature: float = config.DEFAULT_TEMPERATURE,
    genre: str | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    out_dir = output_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(1, num_outputs + 1):
        symbols = generate_sequence(
            num_notes=num_notes,
            temperature=temperature,
            genre=genre,
            humanize=True,
        )
        path = out_dir / f"output_{i}.mid"
        save_midi(symbols, path)
        paths.append(path)
        print(f"Generated {path}")

    return paths


def midi_to_wav(midi_path: Path, wav_path: Path) -> bool:
    """Convert MIDI to WAV using midi2audio + FluidSynth if available."""
    try:
        from midi2audio import FluidSynth

        sf = config.SOUNDFONT_PATH
        if not sf.exists():
            return False
        fs = FluidSynth(str(sf))
        fs.midi_to_audio(str(midi_path), str(wav_path))
        return wav_path.exists()
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate music from trained model")
    parser.add_argument("--notes", type=int, default=config.DEFAULT_NUM_NOTES)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--outputs", type=int, default=config.DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--genre", type=str, default=None, choices=list(config.GENRES))
    parser.add_argument("--to-wav", action="store_true")
    args = parser.parse_args()

    paths = generate_batch(
        num_outputs=args.outputs,
        num_notes=args.notes,
        temperature=args.temperature,
        genre=args.genre,
    )

    if args.to_wav:
        for midi in paths:
            wav = midi.with_suffix(".wav")
            if midi_to_wav(midi, wav):
                print(f"Audio: {wav}")


if __name__ == "__main__":
    main()
