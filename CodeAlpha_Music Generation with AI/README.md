# CodeAlpha — Music Generation with AI

An end-to-end **symbolic music generation** system that trains a deep learning model on MIDI data and generates original piano-style sequences. Built with **PyTorch**, **music21**, and a **Flask** web UI.

## Project Overview

This project learns patterns from MIDI note/chord sequences and predicts the next musical event. It supports:

- **LSTM** (default) and optional **Transformer** architecture
- **Genre conditioning** (classical, jazz, folk)
- **Temperature sampling** for creativity control
- **Humanized** velocities and durations
- **Flask UI** with sliders, downloads, and in-browser audio playback (when FluidSynth is available)

## Architecture

```text
MIDI files -> music21 parsing -> integer sequences -> sliding windows
      -> LSTM/Transformer -> next-token prediction -> MIDI + optional WAV
```

1. **Preprocessing** (`preprocess.py`): extract notes/chords/rests, build vocabulary, create `(input, target)` pairs.
2. **Training** (`train.py`): categorical cross-entropy + Adam, checkpoints, early stopping, loss plot.
3. **Generation** (`generate.py`): seed sequence + autoregressive sampling with temperature.
4. **UI** (`app.py`): generate multiple `output_N.mid` files from the browser.

## Dataset

**Domain:** multi-genre symbolic piano-style MIDI (classical / jazz / folk)

**Sources supported:**

- Synthetic corpus generated locally (default, fast setup)
- Optional MAESTRO subset download via `download_data.py`

### Download / prepare data

```bash
cd "CodeAlpha_Music Generation with AI"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_data.py
```

This ensures **80+ MIDI files** in `data/midi/`.

## Quick Start

### 1. Preprocess

```bash
python preprocess.py
```

Outputs:

- `data/notes.pkl`
- `data/vocab.pkl`

### 2. Train

```bash
# LSTM (default)
python train.py

# Transformer (optional)
python train.py --model transformer
```

Outputs:

- `models/best_model.pth`
- `assets/training_loss.png`

### 3. Generate from CLI

```bash
python generate.py --notes 200 --temperature 0.85 --outputs 4 --genre classical
```

Outputs: `output/output_1.mid` ... `output/output_N.mid`

### 4. Run Web UI

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Configuration

All hyperparameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_TYPE` | `lstm` | `lstm` or `transformer` |
| `SEQUENCE_LENGTH` | 64 | Context window |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 40 | Max epochs |
| `DEFAULT_TEMPERATURE` | 0.85 | Sampling temperature |
| `DEFAULT_NUM_NOTES` | 200 | Generated sequence length |

## Hyperparameters & Model Details

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam (`lr=0.001`)
- **Regularization:** dropout + early stopping
- **Checkpoints:** `models/checkpoints/` + best weights at `models/best_model.pth`

## Audio Playback (optional)

For WAV preview in the UI:

1. Install FluidSynth system package
2. Place a SoundFont at `assets/soundfont/default.sf2`

Without FluidSynth, MIDI download still works.

## Example Questions to Try in UI

- Temperature `0.6` for stable classical phrases
- Temperature `1.1` for more experimental jazz lines
- Genre `folk` with 250 notes
- Generate 5 variations in one run

## Project Structure

```text
CodeAlpha_Music Generation with AI/
├── app.py
├── config.py
├── preprocess.py
├── train.py
├── generate.py
├── download_data.py
├── requirements.txt
├── data/
│   ├── midi/
│   ├── notes.pkl
│   └── vocab.pkl
├── models/
│   ├── architectures.py
│   └── best_model.pth
├── output/
├── assets/
│   └── training_loss.png
├── templates/
├── static/
└── README.md
```

## Known Limitations

- Synthetic/default dataset is smaller than full MAESTRO; musical complexity is moderate.
- Long-range structure is limited by sequence length (64).
- Audio playback depends on optional FluidSynth setup.
- Not intended for commercial music production without larger datasets and longer training.

## Future Improvements

- Train on full MAESTRO / Lakh MIDI datasets
- Add chord-aware loss and meter conditioning
- Export MP3 via ffmpeg pipeline
- Real-time generation streaming in UI
- Fine-tune Music Transformer checkpoints

## License Note

Use only MIDI data you have rights to download and train on. Synthetic data is included for reproducible evaluation.
