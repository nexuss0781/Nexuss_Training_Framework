"""NTF's compact, byte-level causal Transformer architecture (v1)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.projection = nn.Linear(width, width, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).split(width, dim=-1)
        shape = (batch, length, self.heads, self.head_width)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class ByteGPT(nn.Module):
    """Decoder-only Transformer with UTF-8 byte IDs (vocabulary 0..255)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if config.get("vocab_size") != 256:
            raise ValueError("byte tokenizer architecture requires vocab_size=256")
        width = int(config["d_model"])
        heads = int(config["n_heads"])
        layers = int(config["n_layers"])
        max_length = int(config["max_sequence_length"])
        dropout = float(config.get("dropout", 0.0))
        if width < 1 or heads < 1 or layers < 1 or max_length < 2:
            raise ValueError("invalid Transformer dimensions")
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.max_sequence_length = max_length
        self.token_embedding = nn.Embedding(256, width)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 256, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        _, length = token_ids.shape
        if length < 1 or length > self.max_sequence_length:
            raise ValueError("sequence length is outside the configured model context")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))
