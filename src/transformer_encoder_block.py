"""One Transformer encoder block (Post-LayerNorm), NumPy orchestration only.

This module wires existing components into a single block:

    MultiHeadAttention -> residual -> LayerNorm
    -> FeedForward -> residual -> LayerNorm

It does not stack multiple blocks, classify intents, pool sequences, compute
loss, or train parameters.

What one encoder block does
---------------------------
Each token enters as a vector of width ``embedding_dim``. Attention lets tokens
exchange information; the feed-forward network then transforms each token
alone. Residuals keep a copy of the pre-sublayer signal; LayerNorm stabilizes
feature scale. The output has the **same shape** as the input, so blocks can
be stacked later without reshaping.

Why attention runs before the feed-forward network
--------------------------------------------------
Attention is the communication step (tokens gather context). The FFN is the
computation step (each token applies the same MLP). Running attention first
means the MLP sees context-aware vectors, not isolated embeddings.

Why the first residual uses the original block input
----------------------------------------------------
Post-LN attention half::

    y = LayerNorm(x + Attention(x))

Adding ``x`` preserves the pre-attention representation. The first residual
therefore pairs ``inputs`` with ``attention_output``.

Why the second residual uses ``attention_normalized``
-----------------------------------------------------
Post-LN FFN half::

    z = LayerNorm(y + FFN(y))

Here ``y`` is already ``attention_normalized``. The FFN residual must add back
**that** tensor, not the raw block input, so the skip matches what entered the
FFN.

Why two distinct LayerNorm instances are required
-------------------------------------------------
``layer_norm_1`` and ``layer_norm_2`` each own their own ``gamma`` / ``beta``.
Sharing one instance would force both positions to share scale/shift, which a
trained model would not do.

Why every stage keeps ``embedding_dim``
--------------------------------------
Attention projects back to ``embedding_dim``, the FFN expands then returns to
``embedding_dim``, residuals require identical shapes, and LayerNorm is
feature-wise. Shape ``(seq, d)`` is invariant through the block.

Post-LN was selected deliberately
---------------------------------
This file implements only::

    x = LayerNorm(x + Sublayer(x))

for both sublayers. Pre-LN (``x = x + Sublayer(LayerNorm(x))``) is not mixed
in. Residuals stay a pure add so either style could reuse them elsewhere; this
block commits to Post-LN.

Parameters are still random
---------------------------
All weights come from the child modules' random initializers. No classification
or training happens here. Stacking many blocks is a later phase.

Attention mask
--------------
The mask is passed only into ``MultiHeadAttention``. PAD keys get zero
attention weight; sequence length is unchanged. PAD rows are not zeroed by
hand, and the FFN does not see the mask.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from feed_forward import FeedForward
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from residual_connection import add_residual

_DTYPE = np.float32


class TransformerEncoderBlock:
    """Single Post-LN Transformer encoder block.

    Parameters
    ----------
    embedding_dim:
        Token feature width (input and output).
    num_heads:
        Attention heads; must divide ``embedding_dim``.
    hidden_dim:
        Feed-forward hidden width.
    epsilon:
        LayerNorm stability constant for both norms.
    seed:
        Optional base seed. Multi-head attention uses ``seed``; the
        feed-forward network uses ``None`` when ``seed is None``, otherwise
        ``seed + 1``, so the two modules get distinct but reproducible
        initializations.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        epsilon: float = 1e-5,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(embedding_dim, (int, np.integer)) or int(embedding_dim) < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(num_heads, (int, np.integer)) or int(num_heads) < 1:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(hidden_dim, (int, np.integer)) or int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if int(embedding_dim) % int(num_heads) != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not isinstance(epsilon, (int, float, np.floating)) or float(epsilon) <= 0:
            raise ValueError("epsilon must be positive")

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)

        attention_seed = seed
        feed_forward_seed = None if seed is None else seed + 1

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            seed=attention_seed,
        )
        self.feed_forward = FeedForward(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            seed=feed_forward_seed,
        )
        # Separate norms: each position has its own gamma and beta.
        self.layer_norm_1 = LayerNormalization(
            embedding_dim=self.embedding_dim, epsilon=self.epsilon
        )
        self.layer_norm_2 = LayerNormalization(
            embedding_dim=self.embedding_dim, epsilon=self.epsilon
        )

    def forward(
        self,
        inputs: np.ndarray,
        attention_mask: Optional[Union[np.ndarray, list]] = None,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run one Post-LN encoder block.

        Flow::

            attention_output      = MHA(inputs, mask)
            attention_residual    = inputs + attention_output
            attention_normalized  = LN1(attention_residual)
            feed_forward_output   = FFN(attention_normalized)
            feed_forward_residual = attention_normalized + feed_forward_output
            encoder_output        = LN2(feed_forward_residual)

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            attention_mask: Optional length-``sequence_length`` mask of 0/1.
                Forwarded only to multi-head attention.
            return_intermediates: When True, return a dict of named tensors
                for tests and inspection.

        Returns:
            ``encoder_output`` array, or a dict with keys
            ``attention_output``, ``attention_residual``,
            ``attention_normalized``, ``feed_forward_output``,
            ``feed_forward_residual``, ``encoder_output``.
        """
        inputs = self._validate_inputs(inputs)
        if attention_mask is not None:
            self._validate_attention_mask(attention_mask, inputs.shape[0])

        # --- Sublayer 1: multi-head self-attention (Post-LN) ---
        attention_output = self.multi_head_attention.forward(
            inputs, attention_mask=attention_mask
        )
        attention_residual = add_residual(inputs, attention_output)
        attention_normalized = self.layer_norm_1.forward(attention_residual)

        # --- Sublayer 2: position-wise feed-forward (Post-LN) ---
        feed_forward_output = self.feed_forward.forward(attention_normalized)
        feed_forward_residual = add_residual(
            attention_normalized, feed_forward_output
        )
        encoder_output = self.layer_norm_2.forward(feed_forward_residual)

        if return_intermediates:
            return {
                "attention_output": attention_output,
                "attention_residual": attention_residual,
                "attention_normalized": attention_normalized,
                "feed_forward_output": feed_forward_output,
                "feed_forward_residual": feed_forward_residual,
                "encoder_output": encoder_output,
            }
        return encoder_output

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        if not isinstance(inputs, np.ndarray):
            raise ValueError(
                f"inputs must be a NumPy ndarray, got {type(inputs).__name__}"
            )
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2-D (sequence_length, embedding_dim), "
                f"got shape {inputs.shape}"
            )
        sequence_length, width = inputs.shape
        if sequence_length == 0:
            raise ValueError("inputs must contain at least one token row")
        if width == 0:
            raise ValueError("inputs must contain at least one embedding feature")
        if width != self.embedding_dim:
            raise ValueError(
                f"input embedding dimension {width} does not match "
                f"embedding_dim={self.embedding_dim}"
            )
        if np.isnan(inputs).any():
            raise ValueError("inputs must not contain NaN values")
        if np.isinf(inputs).any():
            raise ValueError("inputs must not contain infinite values")
        return inputs.astype(_DTYPE, copy=False)

    def _validate_attention_mask(
        self,
        attention_mask: Union[np.ndarray, list],
        sequence_length: int,
    ) -> None:
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
