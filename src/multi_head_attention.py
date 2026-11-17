"""Multi-head self-attention (NumPy).

This module runs several independent attention heads on the same sequence,
concatenates their outputs, and projects back to ``embedding_dim``. It does
not implement a feed-forward network, residual connections, layer norm, an
encoder block, a classifier, or training.

Why multiple attention heads
----------------------------
One head learns one way of relating tokens (for example nearby words, or a
token and a modifier). Several heads can attend in parallel, so the model can
capture more than one pattern at once. In this educational code the heads are
*not* given manual roles such as "syntax head" or "ecommerce head". Any useful
specialization appears only after a later training loop updates the weights.

Why embedding_dim must be divisible by num_heads
------------------------------------------------
Each head works in a slice of width ``head_dim = embedding_dim // num_heads``.
Splitting and concatenating only line up when the width divides evenly. For
example ``64 / 4 = 16``; ``64 / 5`` would leave a remainder and break the
reshape.

Why scaling uses sqrt(head_dim), not sqrt(embedding_dim)
--------------------------------------------------------
Each head's Q and K have width ``head_dim``, so the dot-product variance
scales with ``head_dim``. The Transformer paper therefore divides by
``sqrt(head_dim)`` inside each head. Using ``sqrt(embedding_dim)`` would
over-shrink the scores when there are many heads.

Why head outputs are concatenated, then mixed with W_output
-------------------------------------------------------------
After attention, each head returns ``(sequence_length, head_dim)``. Stacking
heads side by side restores width ``embedding_dim``. ``W_output`` then mixes
information across heads so later layers see one fused vector per token, not
``num_heads`` isolated channels.

Why PAD positions are masked before softmax
-------------------------------------------
Padded slots are not words. Filling their key-side scores with a large
negative number drives those softmax weights to (nearly) zero on every head,
without changing tensor shapes.

Random projections are not trained yet
--------------------------------------
``W_query``, ``W_key``, ``W_value``, and ``W_output`` start as small random
matrices. They do not yet represent meaningful relationships. Gradients in a
later training phase will move them.

Pipeline position
-----------------
TokenEmbedding + PositionalEncoding
    -> MultiHeadAttention.forward
    -> (later) FFN / encoder block
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

_DTYPE = np.float32
_MASK_FILL = np.float32(-1e9)


class MultiHeadAttention:
    """Several parallel scaled dot-product self-attention heads.

    Parameters
    ----------
    embedding_dim:
        Width of each input and output token vector.
    num_heads:
        Number of parallel heads. Must divide ``embedding_dim``.
    seed:
        Optional RNG seed for reproducible random projections.

    Attributes
    ----------
    head_dim:
        ``embedding_dim // num_heads``, the width of one head's Q/K/V.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        seed: Optional[int] = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        rng = np.random.default_rng(seed)
        scale = 0.02
        # Full-width projections; split_heads later carves out each head.
        self.W_query = rng.normal(
            0.0, scale, size=(embedding_dim, embedding_dim)
        ).astype(_DTYPE, copy=False)
        self.W_key = rng.normal(
            0.0, scale, size=(embedding_dim, embedding_dim)
        ).astype(_DTYPE, copy=False)
        self.W_value = rng.normal(
            0.0, scale, size=(embedding_dim, embedding_dim)
        ).astype(_DTYPE, copy=False)
        self.W_output = rng.normal(
            0.0, scale, size=(embedding_dim, embedding_dim)
        ).astype(_DTYPE, copy=False)

    def project(
        self, inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map token vectors into full-width Q, K, and V.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            ``(Q, K, V)``, each of shape ``(sequence_length, embedding_dim)``.
            Heads are not separated yet; call ``split_heads`` next.
        """
        inputs = self._validate_inputs(inputs)
        query = inputs @ self.W_query
        key = inputs @ self.W_key
        value = inputs @ self.W_value
        return query, key, value

    def split_heads(self, matrix: np.ndarray) -> np.ndarray:
        """Reshape ``(seq, embedding_dim)`` into ``(num_heads, seq, head_dim)``.

        The embedding axis is viewed as ``num_heads`` contiguous blocks of
        width ``head_dim``, then axes are swapped so head is the leading
        batch dimension. Values are not copied into a new order beyond that
        reshape/transpose; head ``h`` owns columns
        ``[h * head_dim : (h + 1) * head_dim]`` of the original matrix.

        Args:
            matrix: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            Shape ``(num_heads, sequence_length, head_dim)``.
        """
        matrix = np.asarray(matrix, dtype=_DTYPE)
        if matrix.ndim != 2 or matrix.shape[1] != self.embedding_dim:
            raise ValueError(
                f"split_heads expects shape (seq, {self.embedding_dim}), "
                f"got {matrix.shape}"
            )

        sequence_length = matrix.shape[0]
        # (seq, embedding_dim) -> (seq, num_heads, head_dim)
        split = matrix.reshape(sequence_length, self.num_heads, self.head_dim)
        # (seq, num_heads, head_dim) -> (num_heads, seq, head_dim)
        return np.transpose(split, (1, 0, 2))

    def calculate_attention(
        self,
        query_heads: np.ndarray,
        key_heads: np.ndarray,
        value_heads: np.ndarray,
        attention_mask: Optional[Union[np.ndarray, list]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run scaled dot-product attention independently on every head.

        For each head::

            scores = Q_head @ K_head.T / sqrt(head_dim)
            weights = softmax(scores)          # over keys, per query row
            output = weights @ V_head

        Args:
            query_heads: Shape ``(num_heads, sequence_length, head_dim)``.
            key_heads: Shape ``(num_heads, sequence_length, head_dim)``.
            value_heads: Shape ``(num_heads, sequence_length, head_dim)``.
            attention_mask: Optional length-``sequence_length`` mask of 0/1.

        Returns:
            ``head_outputs`` with shape
            ``(num_heads, sequence_length, head_dim)`` and
            ``attention_weights`` with shape
            ``(num_heads, sequence_length, sequence_length)``.
        """
        query_heads = np.asarray(query_heads, dtype=_DTYPE)
        key_heads = np.asarray(key_heads, dtype=_DTYPE)
        value_heads = np.asarray(value_heads, dtype=_DTYPE)

        # (h, seq, d) @ (h, d, seq) -> (h, seq, seq)
        scores = query_heads @ np.transpose(key_heads, (0, 2, 1))
        scores = scores / np.sqrt(np.float32(self.head_dim))

        sequence_length = scores.shape[-1]
        if attention_mask is not None:
            mask = self._validate_attention_mask(attention_mask, sequence_length)
            # Broadcast over heads and query rows; suppress PAD key columns.
            pad_keys = mask == 0  # shape (seq,)
            scores = scores.copy()
            scores[:, :, pad_keys] = _MASK_FILL

        attention_weights = self._softmax_last_axis(scores)
        # (h, seq, seq) @ (h, seq, d) -> (h, seq, d)
        head_outputs = attention_weights @ value_heads
        return head_outputs, attention_weights

    def combine_heads(self, head_outputs: np.ndarray) -> np.ndarray:
        """Undo ``split_heads``: ``(num_heads, seq, head_dim)`` -> ``(seq, d)``.

        Args:
            head_outputs: Shape ``(num_heads, sequence_length, head_dim)``.

        Returns:
            Shape ``(sequence_length, embedding_dim)``, with head blocks laid
            out side by side along the feature axis.
        """
        head_outputs = np.asarray(head_outputs, dtype=_DTYPE)
        if head_outputs.ndim != 3:
            raise ValueError(
                f"combine_heads expects 3-D (num_heads, seq, head_dim), "
                f"got shape {head_outputs.shape}"
            )
        if (
            head_outputs.shape[0] != self.num_heads
            or head_outputs.shape[2] != self.head_dim
        ):
            raise ValueError(
                f"combine_heads expects (num_heads, seq, head_dim) = "
                f"({self.num_heads}, *, {self.head_dim}), got {head_outputs.shape}"
            )

        sequence_length = head_outputs.shape[1]
        # (num_heads, seq, head_dim) -> (seq, num_heads, head_dim)
        merged = np.transpose(head_outputs, (1, 0, 2))
        # (seq, num_heads, head_dim) -> (seq, embedding_dim)
        return merged.reshape(sequence_length, self.embedding_dim)

    def forward(
        self,
        inputs: np.ndarray,
        attention_mask: Optional[Union[np.ndarray, list]] = None,
        return_attention: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Full multi-head self-attention on one sequence.

        Flow::

            inputs
              -> Q, K, V
              -> split into heads
              -> scaled attention per head (+ optional PAD mask)
              -> concatenate heads
              -> combined @ W_output

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            attention_mask: Optional ``[1, ..., 0, 0]`` mask.
            return_attention: When True, also return per-head weight tensors.

        Returns:
            ``output`` of shape ``(sequence_length, embedding_dim)``, or
            ``(output, attention_weights)`` when requested. Weights have shape
            ``(num_heads, sequence_length, sequence_length)``.
        """
        query, key, value = self.project(inputs)
        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(value)

        head_outputs, attention_weights = self.calculate_attention(
            query_heads,
            key_heads,
            value_heads,
            attention_mask=attention_mask,
        )
        combined = self.combine_heads(head_outputs)
        output = combined @ self.W_output

        if return_attention:
            return output, attention_weights
        return output

    @staticmethod
    def _softmax_last_axis(scores: np.ndarray) -> np.ndarray:
        """Stable softmax over the key axis for every head and query row."""
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=_DTYPE)
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2-D (sequence_length, embedding_dim), "
                f"got shape {inputs.shape}"
            )
        sequence_length, width = inputs.shape
        if width != self.embedding_dim:
            raise ValueError(
                f"input embedding dimension {width} does not match "
                f"embedding_dim={self.embedding_dim}"
            )
        if sequence_length == 0:
            raise ValueError("inputs must contain at least one token")
        return inputs

    def _validate_attention_mask(
        self,
        attention_mask: Union[np.ndarray, list],
        sequence_length: int,
    ) -> np.ndarray:
        mask = np.asarray(attention_mask)
        if mask.ndim != 1:
            raise ValueError(
                f"attention_mask must be 1-D, got shape {mask.shape}"
            )
        if mask.shape[0] != sequence_length:
            raise ValueError(
                f"attention_mask length {mask.shape[0]} does not match "
                f"sequence length {sequence_length}"
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "attention_mask must contain only 0 and 1 values"
            )
        if np.all(mask == 0):
            raise ValueError(
                "attention_mask is completely masked; at least one real "
                "token (1) is required"
            )
        return mask.astype(np.int64, copy=False)
