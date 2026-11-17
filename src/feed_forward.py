"""Position-wise feed-forward network for a Transformer encoder (NumPy).

This module applies the same two-layer MLP to every token vector. It does not
implement residual connections, layer normalization, an encoder block, a
classifier, or training.

Why attention alone is not enough
---------------------------------
Multi-head attention mixes information *across* positions: each token becomes
a weighted blend of values from the sequence. That is a communication step.
It does not, by itself, apply a rich nonlinear transform to what each token
now contains. The feed-forward network is the *computation* step that follows.

Attention vs feed-forward
-------------------------
- Attention: tokens talk to each other (rows interact through Q/K/V scores).
- Feed-forward: each token is transformed alone. Row i never reads row j.
  The same W1, b1, W2, and b2 are reused at every position.

Why hidden_dim is commonly larger than embedding_dim
----------------------------------------------------
The first layer expands the representation (for example 8 -> 32). Extra width
gives the nonlinearity room to form features before the second layer compresses
back. The classic Transformer uses ``hidden_dim = 4 * embedding_dim``.

Why a nonlinearity sits between the two linear layers
-----------------------------------------------------
Two linear maps in a row collapse into one linear map
(``x @ W1 @ W2 + ...``). ReLU (or another activation) breaks that collapse so
the block can model nonlinear functions of each token.

Why the output width returns to embedding_dim
---------------------------------------------
Later encoder wiring (not built yet) typically adds the block's output to its
input (a residual). Those two tensors must share shape
``(sequence_length, embedding_dim)``. Returning to width 8 in the walkthrough
below is what makes that future addition legal.

Random parameters are not trained yet
-------------------------------------
W1, b1, W2, and b2 start as small random values. They become meaningful only
after a later training loop updates them.

Pipeline position
-----------------
MultiHeadAttention.forward
    -> FeedForward.forward   (this module)
    -> (later) residual + layer norm + encoder block
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

_DTYPE = np.float32


class FeedForward:
    """Two-layer position-wise MLP used inside a Transformer encoder.

    For an input of shape ``(sequence_length, embedding_dim)``, each row is
    processed independently::

        h = ReLU(x @ W1 + b1)      # (seq, hidden_dim)
        y = h @ W2 + b2            # (seq, embedding_dim)

    Parameters
    ----------
    embedding_dim:
        Width of each token vector (input and output).
    hidden_dim:
        Expanded width between the two linear layers.
    seed:
        Optional RNG seed for reproducible initialization.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(embedding_dim, (int, np.integer)) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        rng = np.random.default_rng(seed)
        scale = 0.02
        # (d_model, d_ff) expands each token; bias matches the hidden width.
        self.W1 = rng.normal(
            0.0, scale, size=(self.embedding_dim, self.hidden_dim)
        ).astype(_DTYPE, copy=False)
        self.b1 = rng.normal(0.0, scale, size=(self.hidden_dim,)).astype(
            _DTYPE, copy=False
        )
        # (d_ff, d_model) projects back so residual addition can work later.
        self.W2 = rng.normal(
            0.0, scale, size=(self.hidden_dim, self.embedding_dim)
        ).astype(_DTYPE, copy=False)
        self.b2 = rng.normal(0.0, scale, size=(self.embedding_dim,)).astype(
            _DTYPE, copy=False
        )

    def first_linear(self, inputs: np.ndarray) -> np.ndarray:
        """Expand each token: ``inputs @ W1 + b1``.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.

        Returns:
            Shape ``(sequence_length, hidden_dim)``, still one row per token.
            No activation is applied here.
        """
        inputs = self._validate_inputs(inputs)
        # (seq, d) @ (d, h) -> (seq, h); b1 broadcasts across sequence rows.
        return inputs @ self.W1 + self.b1

    def relu(self, values: np.ndarray) -> np.ndarray:
        """Element-wise ``max(0, x)`` without modifying ``values`` in place.

        Args:
            values: Any NumPy array (typically the first linear output).

        Returns:
            A new array of the same shape. Negatives become 0; positives stay.
        """
        values = np.asarray(values)
        # np.maximum allocates a new array, so the caller's buffer is untouched.
        return np.maximum(values, 0).astype(_DTYPE, copy=False)

    def second_linear(self, hidden_values: np.ndarray) -> np.ndarray:
        """Project back to embedding width: ``hidden @ W2 + b2``.

        Args:
            hidden_values: Shape ``(sequence_length, hidden_dim)``.

        Returns:
            Shape ``(sequence_length, embedding_dim)``.
        """
        hidden_values = np.asarray(hidden_values, dtype=_DTYPE)
        if hidden_values.ndim != 2:
            raise ValueError(
                f"hidden_values must be 2-D (sequence_length, hidden_dim), "
                f"got shape {hidden_values.shape}"
            )
        if hidden_values.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden width {hidden_values.shape[1]} does not match "
                f"hidden_dim={self.hidden_dim}"
            )
        # (seq, h) @ (h, d) -> (seq, d)
        return hidden_values @ self.W2 + self.b2

    def forward(
        self,
        inputs: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Apply the position-wise FFN to every token row.

        Flow::

            inputs
              -> first_linear   (seq, hidden_dim)
              -> relu
              -> second_linear  (seq, embedding_dim)

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            return_intermediates: When True, also return the pre- and
                post-ReLU hidden matrices for inspection and tests.

        Returns:
            ``output`` alone, or
            ``(output, hidden_pre_activation, hidden_after_activation)``.
        """
        hidden_pre_activation = self.first_linear(inputs)
        hidden_after_activation = self.relu(hidden_pre_activation)
        output = self.second_linear(hidden_after_activation)

        if return_intermediates:
            return output, hidden_pre_activation, hidden_after_activation
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
            raise ValueError("inputs must contain at least one token")
        if width != self.embedding_dim:
            raise ValueError(
                f"input embedding dimension {width} does not match "
                f"embedding_dim={self.embedding_dim}"
            )
        return inputs
