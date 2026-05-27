"""Central hyperparameters and paths for AI music generation."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
MIDI_DIR = BASE_DIR / "data" / "midi"
DATA_DIR = BASE_DIR / "data"
NOTES_PKL = DATA_DIR / "notes.pkl"
VOCAB_PKL = DATA_DIR / "vocab.pkl"
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
LOSS_PLOT_PATH = ASSETS_DIR / "training_loss.png"
SOUNDFONT_PATH = BASE_DIR / "assets" / "soundfont" / "default.sf2"

# Dataset
MIN_MIDI_FILES = 60
TARGET_MIDI_FILES = 80
GENRES = ("classical", "jazz", "folk")

# Preprocessing
SEQUENCE_LENGTH = 64
STEP_SIZE = 1
MIN_NOTES_PER_FILE = 20

# Model selection: "lstm" or "transformer"
MODEL_TYPE = "lstm"

# LSTM architecture
EMBEDDING_DIM = 128
LSTM_UNITS = 256
LSTM_LAYERS = 2
DROPOUT = 0.3

# Transformer architecture (optional)
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.2

# Training
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 6
CHECKPOINT_EVERY = 5

# Generation
DEFAULT_NUM_NOTES = 200
DEFAULT_TEMPERATURE = 0.85
DEFAULT_NUM_OUTPUTS = 4
MAX_NUM_NOTES = 500
MIN_TEMPERATURE = 0.3
MAX_TEMPERATURE = 1.5

# Humanization (velocity/duration jitter)
HUMANIZE_VELOCITY_RANGE = 8
HUMANIZE_DURATION_RANGE = 0.08

# Flask UI
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
