"""Single-head scaled dot-product self-attention (NumPy).

This module lets each token in a sequence gather information from the other
tokens. It does not implement multi-head attention, a feed-forward network,
a full encoder block, a classifier, or training. Those come later.

What Query, Key, and Value represent
------------------------------------
Think of each token as asking a question and offering an answer:

- Query (Q): what this token is looking for.
- Key (K): what this token advertises about itself.
- Value (V): the content that will be mixed into other tokens.

A high score between query_i and key_j means token i should pull more of
token j's value into its new representation.

Why Q, K, and V come from the same input
----------------------------------------
In *self*-attention every token plays all three roles. The same sequence is
projected three ways with W_query, W_key, and W_value. That is different from
cross-attention, where queries come from one sequence and keys/values from
another (for example decoder attending to encoder outputs).

Why scores are divided by sqrt(attention_dim)
---------------------------------------------
``Q @ K.T`` grows with the width of Q and K. Large raw scores push softmax
into a very peaky region where gradients vanish. Dividing by
``sqrt(attention_dim)`` (Vaswani et al., 2017) keeps the variance of the
scores roughly stable as the width changes.

Why softmax is row-wise
-----------------------
Row i is token i's distribution over all tokens as keys. Softmax turns that
row into non-negative weights that sum to 1, so token i takes a weighted
average of the value vectors.

Why PAD tokens must be masked before softmax
--------------------------------------------
Padded positions are not real words. If they stay in the score matrix,
softmax can still give them probability mass. Setting their key-side scores
to a large negative number before softmax drives those probabilities to
(nearly) zero without changing the matrix shape.

Learnable projections, not trained yet
--------------------------------------
W_query, W_key, and W_value start as small random matrices. They only become
meaningful after a later training loop updates them with gradients. This file
only runs the forward computation.

Pipeline position
-----------------
TokenEmbedding + PositionalEncoding
    -> SelfAttention.forward(inputs, attention_mask)
    -> (later) multi-head / FFN / encoder block
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

_DTYPE = np.float32

# Softmax(x) for x around -1e9 is effectively 0 in float32.
_MASK_FILL = np.float32(-1e9)


class SelfAttention:
    """One head of scaled dot-product self-attention.

    Parameters
    ----------
    embedding_dim:
        Width of each input token vector (must match the embedding / PE size).
    attention_dim:
        Width of Q, K, and V. Defaults to ``embedding_dim`` when omitted.
    seed:
        Optional RNG seed so tests and demos are reproducible.

    Notes
    -----
    Output width is ``attention_dim``, not ``embedding_dim``. A later multi-head
    block will usually concatenate heads and project back to ``embedding_dim``.
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        if attention_dim is None:
            attention_dim = embedding_dim
        if attention_dim < 1:
            raise ValueError("attention_dim must be at least 1")

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

        rng = np.random.default_rng(seed)
        # Small Gaussian init, same spirit as the token embedding table.
        scale = 0.02
        self.W_query = rng.normal(
            0.0, scale, size=(embedding_dim, attention_dim)
        ).astype(_DTYPE, copy=False)
        self.W_key = rng.normal(
            0.0, scale, size=(embedding_dim, attention_dim)
        ).astype(_DTYPE, copy=False)
        self.W_value = rng.normal(
            0.0, scale, size=(embedding_dim, attention_dim)
        ).astype(_DTYPE, copy=False)

    def project(
        self, inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map token vectors into query, key, and value spaces.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            ``(Q, K, V)``, each of shape
            ``(sequence_length, attention_dim)``.

        Notes:
            ``Q = inputs @ W_query`` (and likewise for K and V). Each token
            keeps its own row; only the feature width changes from
            ``embedding_dim`` to ``attention_dim``.
        """
        inputs = self._validate_inputs(inputs)
        # (seq, d_model) @ (d_model, d_k) -> (seq, d_k)
        query = inputs @ self.W_query
        key = inputs @ self.W_key
        value = inputs @ self.W_value
        return query, key, value

    def calculate_scores(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Compute scaled attention scores ``(Q @ K.T) / sqrt(d_k)``.

        Scaling by ``sqrt(attention_dim)`` stops the dot products from growing
        with width, which would otherwise make softmax saturating and hard to
        train later.

        Args:
            query: Shape ``(sequence_length, attention_dim)``.
            key: Shape ``(sequence_length, attention_dim)``.

        Returns:
            Shape ``(sequence_length, sequence_length)``. Entry ``[i, j]`` is
            how strongly token ``i`` (query) matches token ``j`` (key), before
            softmax.
        """
        query = np.asarray(query, dtype=_DTYPE)
        key = np.asarray(key, dtype=_DTYPE)
        # (seq, d_k) @ (d_k, seq) -> (seq, seq)
        scores = query @ key.T
        scale = np.sqrt(np.float32(self.attention_dim))
        return scores / scale

    def apply_padding_mask(
        self,
        scores: np.ndarray,
        attention_mask: Union[np.ndarray, list],
    ) -> np.ndarray:
        """Zero out PAD keys by filling their columns with a large negative.

        Example mask ``[1, 1, 1, 1, 0, 0]`` means the last two positions are
        PAD. After masking, every row's scores for those columns are ~-1e9, so
        softmax puts ~0 weight on them. Rows (query positions) are not removed;
        the matrix shape stays ``(seq, seq)``.

        Args:
            scores: Shape ``(sequence_length, sequence_length)``.
            attention_mask: Length ``sequence_length``, values only ``0`` or ``1``.

        Returns:
            A copy of ``scores`` with PAD key columns suppressed.

        Raises:
            ValueError: Mask length, values, or "all PAD" cases fail validation.
        """
        scores = np.asarray(scores, dtype=_DTYPE)
        mask = self._validate_attention_mask(attention_mask, scores.shape[0])

        masked = scores.copy()
        # Key axis is columns: wherever the mask is 0, block that whole column.
        pad_columns = mask == 0
        masked[:, pad_columns] = _MASK_FILL
        return masked

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """Row-wise numerically stable softmax.

        Subtracting each row's maximum before ``exp`` avoids overflow when a
        score is large, without changing the resulting probabilities.

        Args:
            scores: Shape ``(sequence_length, sequence_length)``.

        Returns:
            Same shape. Each row sums to approximately 1.
        """
        scores = np.asarray(scores, dtype=_DTYPE)
        # Stable softmax: exp(x - max) / sum(exp(x - max)).
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        # Rows that are entirely -inf would divide by zero; callers should not
        # produce that after validation, but guard anyway.
        row_sums = np.sum(exponentials, axis=1, keepdims=True)
        return exponentials / row_sums

    def forward(
        self,
        inputs: np.ndarray,
        attention_mask: Optional[Union[np.ndarray, list]] = None,
        return_attention: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Run full single-head self-attention on one sequence.

        Flow::

            inputs
              -> Q, K, V
              -> Q @ K.T / sqrt(d_k)
              -> optional PAD mask on key columns
              -> row-wise softmax
              -> attention_weights @ V

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            attention_mask: Optional ``[1, 1, ..., 0, 0]`` mask aligned with
                the sequence. ``None`` means every position is real.
            return_attention: When True, also return the weight matrix for
                inspection and tests.

        Returns:
            ``output`` of shape ``(sequence_length, attention_dim)``, or
            ``(output, attention_weights)`` when ``return_attention`` is True.
        """
        query, key, value = self.project(inputs)
        scores = self.calculate_scores(query, key)

        if attention_mask is not None:
            scores = self.apply_padding_mask(scores, attention_mask)

        attention_weights = self.softmax(scores)
        # (seq, seq) @ (seq, d_k) -> (seq, d_k): each token is a mix of values.
        output = attention_weights @ value

        if return_attention:
            return output, attention_weights
        return output

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Require a 2-D sequence whose width matches ``embedding_dim``."""
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
        """Require a 0/1 mask of length ``sequence_length`` with some real tokens."""
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
        # Allow bool or int 0/1, reject anything else.
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
