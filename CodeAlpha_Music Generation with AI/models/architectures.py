"""Neural network architectures for symbolic music generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """Stacked LSTM with genre conditioning for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_genres: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, genre_ids: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(x)
        if genre_ids is not None:
            g = self.genre_embedding(genre_ids).unsqueeze(1).expand(-1, emb.size(1), -1)
            emb = emb + g * 0.25
        out, _ = self.lstm(emb)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    """Lightweight Transformer encoder for symbolic music."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_genres: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, genre_ids: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(x)
        if genre_ids is not None:
            g = self.genre_embedding(genre_ids).unsqueeze(1)
            emb = emb + g * 0.25
        emb = self.pos_encoder(emb)
        mask = self._causal_mask(x.size(1), x.device)
        out = self.transformer(emb, mask=mask)
        return self.fc(out[:, -1, :])

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()


def build_model(
    model_type: str,
    vocab_size: int,
    num_genres: int,
    config_module,
) -> nn.Module:
    if model_type == "transformer":
        return MusicTransformer(
            vocab_size=vocab_size,
            d_model=config_module.TRANSFORMER_DIM,
            nhead=config_module.TRANSFORMER_HEADS,
            num_layers=config_module.TRANSFORMER_LAYERS,
            dim_feedforward=config_module.TRANSFORMER_FF_DIM,
            dropout=config_module.TRANSFORMER_DROPOUT,
            num_genres=num_genres,
        )
    return MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=config_module.EMBEDDING_DIM,
        hidden_dim=config_module.LSTM_UNITS,
        num_layers=config_module.LSTM_LAYERS,
        dropout=config_module.DROPOUT,
        num_genres=num_genres,
    )
