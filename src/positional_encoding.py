"""Sinusoidal positional encoding for the intent-classification transformer.

This module adds word-order information to token embeddings. It does not
implement attention or any transformer block. Attention will later run on the
sum of token vectors and these position vectors.

Why transformers need positional encoding
-----------------------------------------
Self-attention looks at a set of vectors. Without extra information, the same
set yields the same result no matter how you shuffle it. "nike shoes under
5000" and "5000 under shoes nike" would look identical to the model. Intent
classification needs order: "under 5000" is not the same as "5000 under".

Why embeddings alone are insufficient
-------------------------------------
TokenEmbedding maps each id to a meaning vector. Every time the id "nike"
appears, it gets the same row from the table. That row says what the word is,
not where it sits in the sentence. Position has to be supplied separately.

Token meaning vs token position
-------------------------------
- Meaning (token embedding): "shoes" is footwear-related.
- Position (this module): "shoes" is the third token in this query.

The usual teaching approach, taken from Vaswani et al. 2017, is to *add*
the two vectors. After addition, each position carries both signals in one
array of shape (sequence_length, embedding_dim).

Sinusoidal encodings
--------------------
For position ``pos`` and dimension index ``i``:

    PE(pos, 2i)   = sin(pos / 10000^(2i / embedding_dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / embedding_dim))

Even columns use sine, odd columns use cosine. Different frequencies let a
later layer read both absolute position and relative offsets. The table is
fixed (not learned) in this educational version.

Pipeline position
-----------------
TokenEmbedding.forward -> PositionalEncoding.add_position_information
    -> (later) attention
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_DTYPE = np.float32


class PositionalEncoding:
    """Fixed sinusoidal position table of shape (max_length, embedding_dim).

    The table is built once in ``__init__`` and reused. ``forward`` only
    slices the first ``sequence_length`` rows. That keeps every call cheap
    and makes position 0 always mean "first token in the padded sequence".

    Attributes:
        max_length: Longest sequence this table can encode (usually the same
            max_length used when padding the dataset).
        embedding_dim: Must match TokenEmbedding.embedding_dim so addition
            is element-wise.
        encodings: Precomputed PE matrix, shape (max_length, embedding_dim).
    """

    def __init__(self, max_length: int, embedding_dim: int) -> None:
        """Build the sinusoidal table for positions ``0 .. max_length-1``.

        Args:
            max_length: Maximum number of positions to support.
            embedding_dim: Width of each position vector. Should equal the
                token embedding size.

        Raises:
            ValueError: max_length or embedding_dim is less than 1.
        """
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.encodings = self._build_sinusoidal_table(max_length, embedding_dim)

    @staticmethod
    def _build_sinusoidal_table(
        max_length: int, embedding_dim: int
    ) -> np.ndarray:
        """Create PE[pos, dim] with the original Transformer formulas.

        We compute the angle rates for every even/odd pair of columns, then
        fill sine into even indices and cosine into odd indices. If
        embedding_dim is odd, the final column only receives the sine term
        for that half-index (there is no matching odd slot).
        """
        # positions: shape (max_length, 1) so it broadcasts against rates.
        positions = np.arange(max_length, dtype=_DTYPE)[:, np.newaxis]

        # i = 0, 1, 2, ... for each pair of dimensions (2i, 2i+1).
        dimension_pairs = np.arange(0, embedding_dim, 2, dtype=_DTYPE)

        # 10000^(2i / d_model) in the denominator from the paper.
        # Written as exp( (2i / d) * -log(10000) ) for numerical stability.
        angle_rates = np.exp(
            dimension_pairs
            * -(np.log(np.float32(10000.0)) / np.float32(embedding_dim))
        )
        angle_rates = angle_rates[np.newaxis, :]  # shape (1, num_pairs)
        angles = positions * angle_rates  # shape (max_length, num_pairs)

        table = np.zeros((max_length, embedding_dim), dtype=_DTYPE)
        table[:, 0::2] = np.sin(angles)

        # Odd columns exist only when there is a matching pair slot.
        odd_width = embedding_dim // 2
        table[:, 1::2] = np.cos(angles[:, :odd_width])

        return table

    def forward(self, sequence_length: int) -> np.ndarray:
        """Return positional vectors for ``0 .. sequence_length-1``.

        Args:
            sequence_length: How many positions to return. Must be between 1
                and ``max_length`` inclusive, or 0 for an empty slice.

        Returns:
            Array of shape ``(sequence_length, embedding_dim)``.

        Raises:
            ValueError: sequence_length is negative or greater than max_length.

        Example:
            sequence_length=5 -> shape (5, embedding_dim)
        """
        if sequence_length < 0:
            raise ValueError("sequence_length must be >= 0")
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length {sequence_length} exceeds max_length "
                f"{self.max_length}"
            )
        return self.encodings[:sequence_length]

    def add_position_information(
        self, token_embeddings: np.ndarray
    ) -> np.ndarray:
        """Add positional encodings to a sequence of token embeddings.

        Args:
            token_embeddings: Array of shape
                ``(sequence_length, embedding_dim)`` from TokenEmbedding.

        Returns:
            ``token_embeddings + PE[:sequence_length]``, same shape as the
            input. Meaning and position are mixed in each row.

        Raises:
            ValueError: Rank or width does not match, or the sequence is
                longer than ``max_length``.
        """
        embeddings = np.asarray(token_embeddings, dtype=_DTYPE)
        if embeddings.ndim != 2:
            raise ValueError(
                f"token_embeddings must be 2-D, got shape {embeddings.shape}"
            )

        sequence_length, width = embeddings.shape
        if width != self.embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: got {width}, expected "
                f"{self.embedding_dim}"
            )
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length {sequence_length} exceeds max_length "
                f"{self.max_length}"
            )

        return embeddings + self.forward(sequence_length)

    def encoding_shape(self) -> Tuple[int, int]:
        """Return ``(max_length, embedding_dim)`` for the full table."""
        return (self.max_length, self.embedding_dim)
