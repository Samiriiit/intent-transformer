"""Trainable intent-classification Transformer (PyTorch).

This module is the first *trainable* model in the project. The earlier NumPy
files stay untouched as a readable walkthrough of embeddings, attention,
residuals, LayerNorm, and Post-LN encoder stacks. They do not compute
gradients.

Why switch to PyTorch here
--------------------------
NumPy made every matrix multiply explicit. Training needs automatic
differentiation: PyTorch records ops on tensors and ``loss.backward()`` fills
``.grad`` on parameters. We do not hand-write NumPy backprop.

Intent classes
--------------
GENERIC = 0, ECOMMERCE = 1, RETAIL = 2, LEGAL = 3 (``num_classes = 4``).

Architecture
------------
``input_ids`` → Embedding → ×√d → + sinusoidal PE → Dropout
→ ``nn.TransformerEncoder`` (Post-LN, ``norm_first=False``)
→ mask-aware mean pool → Linear → **logits** ``(batch, 4)``.

Why return logits (not softmax) from ``forward``
------------------------------------------------
``torch.nn.CrossEntropyLoss`` expects raw scores and applies log-softmax
internally. Softmax in ``forward`` would be wrong for that loss and can hurt
numerical stability. Use ``predict_proba`` when you need probabilities.

Why PAD is excluded from mean pooling
-------------------------------------
Padded slots are not words. Averaging them would dilute the sentence vector
with empty positions. The attention mask (1 = real, 0 = PAD) zeros those
rows before the sum/count.

Why the padding-mask convention is inverted for ``TransformerEncoder``
----------------------------------------------------------------------
Our mask uses 1 = keep, 0 = pad. PyTorch ``src_key_padding_mask`` uses
``True`` = ignore. Hence ``src_key_padding_mask = (attention_mask == 0)``.

Why ``norm_first=False``
------------------------
That is Post-LayerNorm (``x = Norm(x + Sublayer(x))``), matching the educational
NumPy encoder block. ``norm_first=True`` would be Pre-LN.

Pipeline note
-------------
No training loop, optimizer, or dataset I/O lives here—only the module and
forward/predict_proba.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed label ids (same mapping as the NumPy dataset module).
GENERIC = 0
ECOMMERCE = 1
RETAIL = 2
LEGAL = 3


def _sinusoidal_positional_encoding(
    max_length: int, embedding_dim: int
) -> torch.Tensor:
    """Build fixed PE with shape ``(1, max_length, embedding_dim)``."""
    position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    dimension_pairs = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
    angle_rates = torch.exp(
        dimension_pairs * (-math.log(10000.0) / embedding_dim)
    )
    angles = position * angle_rates.unsqueeze(0)

    pe = torch.zeros(max_length, embedding_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : embedding_dim // 2])
    return pe.unsqueeze(0)


class IntentClassificationTransformer(nn.Module):
    """Post-LN Transformer encoder + masked mean pool + linear classifier.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary (including specials).
    embedding_dim:
        Model width ``d_model``. Must be divisible by ``num_heads``.
    num_heads:
        Attention heads in each encoder layer.
    hidden_dim:
        Feed-forward width inside each encoder layer.
    num_layers:
        Number of stacked ``TransformerEncoderLayer`` modules.
    num_classes:
        Intent classes (default 4).
    max_length:
        Maximum sequence length supported by the positional table.
    pad_token_id:
        Id whose embedding is zeroed (``padding_idx``).
    dropout:
        Dropout probability in embeddings/encoder (0 disables).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        max_length: int = 64,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("dropout must be in the closed interval [0, 1]")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.dropout_p = float(dropout)

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        # Fixed sinusoidal table — not a parameter, not updated by optimizers.
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_positional_encoding(max_length, embedding_dim),
            persistent=True,
        )

        self.embedding_dropout = nn.Dropout(self.dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=self.dropout_p,
            activation="relu",
            batch_first=True,
            norm_first=False,  # Post-LN, matching the NumPy encoder block.
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return intent logits of shape ``(batch_size, num_classes)``.

        Args:
            input_ids: ``(batch_size, sequence_length)`` token ids.
            attention_mask: ``(batch_size, sequence_length)`` with 1 = real,
                0 = PAD.
        """
        self._validate_batch(input_ids, attention_mask)

        batch_size, sequence_length = input_ids.shape

        # (B, S) -> (B, S, D)
        token_vectors = self.token_embedding(input_ids)
        token_vectors = token_vectors * math.sqrt(self.embedding_dim)

        # Slice PE to the active length: (1, S, D) broadcasts over batch.
        positions = self.positional_encoding[:, :sequence_length, :]
        hidden = self.embedding_dropout(token_vectors + positions)

        # PyTorch: True = ignore this key position (our 0 = PAD).
        src_key_padding_mask = attention_mask == 0

        encoded = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)

        pooled = self._masked_mean_pool(encoded, attention_mask)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Softmax probabilities with eval + ``no_grad`` (restores train mode)."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask)
                return F.softmax(logits, dim=-1)
        finally:
            self.train(was_training)

    def _masked_mean_pool(
        self, encoded_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean over real tokens only → ``(batch_size, embedding_dim)``."""
        expanded_mask = attention_mask.unsqueeze(-1).to(dtype=encoded_output.dtype)
        masked_output = encoded_output * expanded_mask
        summed = masked_output.sum(dim=1)
        counts = expanded_mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def _validate_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be rank 2 (batch, sequence), got shape "
                f"{tuple(input_ids.shape)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must be rank 2 (batch, sequence), got shape "
                f"{tuple(attention_mask.shape)}"
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} does not match "
                f"attention_mask shape {tuple(attention_mask.shape)}"
            )

        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("sequence_length must be at least 1")
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length {sequence_length} exceeds max_length "
                f"{self.max_length}"
            )

        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0 and 1 values")

        real_counts = attention_mask.sum(dim=1)
        if torch.any(real_counts < 1):
            raise ValueError(
                "each sample must contain at least one non-PAD token"
            )

        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError(
                f"input_ids must be in [0, {self.vocab_size - 1}]"
            )
