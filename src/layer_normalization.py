"""Layer Normalization for Transformer token vectors (NumPy).

This module normalizes each token row across its embedding features. It does
not implement residual connections, an encoder block, a classifier, or
training.

Why Transformer activations benefit from normalization
------------------------------------------------------
After attention and the feed-forward network, feature scales can drift: some
dimensions become very large, others tiny. That makes later layers harder to
optimize. LayerNorm recenters and rescales each token so features have a
stable mean and variance before the next computation.

Row-wise statistics (not sequence-wise, not batch-wise)
------------------------------------------------------
For shape ``(sequence_length, embedding_dim)``:

- Mean and variance are computed **per token row** across the embedding axis.
- Token i's statistics never use token j's values.
- Changing one row must not change another row's normalized output.

That is different from BatchNorm, which estimates statistics across a batch
(and often across positions) for each feature channel. Transformers usually
prefer LayerNorm because sequence lengths and batch makeup vary, while each
token still needs a consistent per-vector normalization.

Why epsilon is added before the square root
-------------------------------------------
Variance can be exactly zero (a constant row). ``sqrt(0)`` would make the
denominator zero. ``sqrt(variance + epsilon)`` keeps the division defined and
numerically stable.

Why gamma starts as ones and beta as zeros
------------------------------------------
``output = gamma * normalized + beta``. With gamma=1 and beta=0, the first
forward pass returns the pure normalized values. Training (later) can move
gamma and beta so each feature learns its own scale and offset. In this file
they stay fixed because there is no optimizer yet.

Why input and output shapes match
---------------------------------
LayerNorm is an element-wise affine transform after normalization. Width stays
``embedding_dim``, so a future residual ``x + LayerNorm(...)`` or
``LayerNorm(x + ...)`` can still align shapes.

Pipeline position
-----------------
Attention / FeedForward
    -> LayerNormalization.forward   (this module)
    -> (later) residual wiring inside an encoder block
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

_DTYPE = np.float32


class LayerNormalization:
    """Normalize each token across embedding features, then scale and shift.

    For one row ``x`` of length ``embedding_dim``::

        mean = mean(x)
        var  = mean((x - mean)^2)          # population variance, ddof=0
        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta

    Parameters
    ----------
    embedding_dim:
        Width of each token vector.
    epsilon:
        Small positive constant added inside the square root.
    """

    def __init__(self, embedding_dim: int, epsilon: float = 1e-5) -> None:
        if not isinstance(embedding_dim, (int, np.integer)) or int(embedding_dim) < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(epsilon, (int, float, np.floating)) or float(epsilon) <= 0:
            raise ValueError("epsilon must be a positive number")

        self.embedding_dim = int(embedding_dim)
        self.epsilon = float(epsilon)

        # Learnable in a full training setup; fixed identity transform for now.
        self.gamma = np.ones((self.embedding_dim,), dtype=_DTYPE)
        self.beta = np.zeros((self.embedding_dim,), dtype=_DTYPE)

    def calculate_mean(self, inputs: np.ndarray) -> np.ndarray:
        """Row-wise mean over the embedding axis.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            Shape ``(sequence_length, 1)`` so subtraction broadcasts per row.
            This is **not** one global mean for the whole matrix.
        """
        inputs = self._validate_inputs(inputs)
        # axis=-1: average across features; keepdims keeps (seq, 1).
        return np.mean(inputs, axis=-1, keepdims=True)

    def calculate_variance(
        self,
        inputs: np.ndarray,
        mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Row-wise population variance over the embedding axis (ddof=0).

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            mean: Optional precomputed mean with shape ``(sequence_length, 1)``.
                When omitted, the mean is computed here.

        Returns:
            Shape ``(sequence_length, 1)``.
        """
        inputs = self._validate_inputs(inputs)
        if mean is None:
            mean = self.calculate_mean(inputs)
        else:
            mean = np.asarray(mean, dtype=_DTYPE)

        # Population variance: mean of squared deviations (NumPy default ddof=0).
        centered = inputs - mean
        return np.mean(centered * centered, axis=-1, keepdims=True)

    def normalize(
        self,
        inputs: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> np.ndarray:
        """Standardize features: ``(x - mean) / sqrt(var + epsilon)``.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            mean: Shape ``(sequence_length, 1)``.
            variance: Shape ``(sequence_length, 1)``.

        Returns:
            Normalized array of the same shape as ``inputs``. The original
            ``inputs`` buffer is not modified.
        """
        inputs = np.asarray(inputs, dtype=_DTYPE)
        mean = np.asarray(mean, dtype=_DTYPE)
        variance = np.asarray(variance, dtype=_DTYPE)
        # New array via arithmetic; caller retains the pre-norm values.
        return (inputs - mean) / np.sqrt(variance + np.float32(self.epsilon))

    def apply_scale_and_shift(self, normalized: np.ndarray) -> np.ndarray:
        """Apply ``gamma * normalized + beta`` with per-feature broadcasting.

        Args:
            normalized: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            Same shape. Every token row reuses the same gamma and beta vectors.
        """
        normalized = np.asarray(normalized, dtype=_DTYPE)
        # gamma/beta are (embedding_dim,); they broadcast over sequence rows.
        return self.gamma * normalized + self.beta

    def forward(
        self,
        inputs: np.ndarray,
        return_statistics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Full LayerNorm: mean, variance, normalize, then gamma/beta.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            return_statistics: When True, also return mean, variance, and the
                pre-affine normalized matrix for tests and inspection.

        Returns:
            ``output``, or
            ``(output, mean, variance, normalized)`` when requested.
        """
        inputs = self._validate_inputs(inputs)
        mean = self.calculate_mean(inputs)
        variance = self.calculate_variance(inputs, mean=mean)
        normalized = self.normalize(inputs, mean, variance)
        output = self.apply_scale_and_shift(normalized)

        if return_statistics:
            return output, mean, variance, normalized
        return output

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=_DTYPE)
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2-D (sequence_length, embedding_dim), "
                f"got shape {inputs.shape}"
            )
        sequence_length, width = inputs.shape
        if sequence_length == 0:
            raise ValueError("inputs must contain at least one token row")
        if width != self.embedding_dim:
            raise ValueError(
                f"input embedding dimension {width} does not match "
                f"embedding_dim={self.embedding_dim}"
            )
        if np.isnan(inputs).any():
            raise ValueError("inputs must not contain NaN values")
        if np.isinf(inputs).any():
            raise ValueError("inputs must not contain infinite values")
        return inputs
