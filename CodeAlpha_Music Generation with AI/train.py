"""Train LSTM or Transformer model on preprocessed MIDI sequences."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import config
from models.architectures import build_model
from preprocess import load_preprocessed, preprocess


class MusicDataset(Dataset):
    def __init__(self, inputs, targets, genres):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.genres = torch.tensor(genres, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.genres[idx], self.targets[idx]


def train_model(force_preprocess: bool = False) -> dict:
    if force_preprocess or not config.NOTES_PKL.exists():
        preprocess(save=True)
    data = load_preprocessed()

    dataset = MusicDataset(data["inputs"], data["targets"], data["genres"])
    val_size = int(len(dataset) * config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.MODEL_TYPE,
        data["vocab_size"],
        data["num_genres"],
        config,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running = 0.0
        for x, g, y in train_loader:
            x, g, y = x.to(device), g.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, g)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        train_losses.append(train_loss)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, g, y in val_loader:
                x, g, y = x.to(device), g.to(device), y.to(device)
                logits = model(x, g)
                val_running += criterion(logits, y).item()
        val_loss = val_running / max(len(val_loader), 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{config.EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": config.MODEL_TYPE,
                    "vocab_size": data["vocab_size"],
                    "num_genres": data["num_genres"],
                    "config": {
                        "sequence_length": config.SEQUENCE_LENGTH,
                        "embedding_dim": config.EMBEDDING_DIM,
                        "lstm_units": config.LSTM_UNITS,
                        "lstm_layers": config.LSTM_LAYERS,
                        "dropout": config.DROPOUT,
                    },
                },
                config.BEST_MODEL_PATH,
            )
            print(f"  Saved best model -> {config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1

        if epoch % config.CHECKPOINT_EVERY == 0:
            ckpt = config.CHECKPOINT_DIR / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Music Generation Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=120)
    plt.close()

    meta = {
        "epochs_run": len(train_losses),
        "best_val_loss": best_val_loss,
        "loss_plot": str(config.LOSS_PLOT_PATH),
    }
    with (config.MODEL_DIR / "training_meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train music generation model")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        default=config.MODEL_TYPE,
        help="Model architecture",
    )
    args = parser.parse_args()
    config.MODEL_TYPE = args.model
    train_model(force_preprocess=args.preprocess)


if __name__ == "__main__":
    main()
