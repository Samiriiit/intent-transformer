"""Residual (skip) connections for Transformer sublayers (NumPy).

A residual connection adds a sublayer's output back to the sublayer's input::

    residual_output = original_input + sublayer_output

What a residual connection is
-----------------------------
Instead of replacing ``x`` with ``Sublayer(x)``, the network keeps both:

    y = x + Sublayer(x)

The original representation is still present in ``y``, mixed with whatever the
sublayer computed (attention, feed-forward, and so on).

Why add the original input back
-------------------------------
Deep stacks of attention and MLPs can wash out early signal. Adding ``x``
gives later layers a **direct path** to earlier representations: gradients and
information can skip the sublayer mathematically. That helps preserve token
identity across depth and makes deep Transformers easier to train once an
optimizer exists.

Shape constraints
-----------------
Both sides must be exactly ``(sequence_length, embedding_dim)``. Addition is
element-wise and position-aligned: token i, feature j on the left matches the
same slot on the right. Sequence length and embedding width therefore do not
change. Broadcasting (for example ``(6, 8) + (1, 8)``) is **rejected** here so
a shape bug cannot hide inside NumPy's broadcast rules.

What this module does *not* do
------------------------------
Only addition. No LayerNorm, activation, attention, or dropout.

LayerNorm placement is intentionally separate. Common Transformer patterns are:

- Post-LN: ``x = LayerNorm(x + Sublayer(x))``
- Pre-LN:  ``x = x + Sublayer(LayerNorm(x))``

The encoder-block phase will choose one. Keeping ``add_residual`` independent
lets either arrangement call this helper unchanged.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float32


def add_residual(
    original_input: np.ndarray,
    sublayer_output: np.ndarray,
) -> np.ndarray:
    """Add a sublayer output to its input element-wise.

    Args:
        original_input: Tensor ``x`` of shape
            ``(sequence_length, embedding_dim)``. Typically the tensor that
            entered attention or the feed-forward network.
        sublayer_output: Tensor ``Sublayer(x)`` with **exactly** the same
            shape as ``original_input``.

    Returns:
        A **new** array ``original_input + sublayer_output`` with the same
        shape. Neither argument is modified in place.

    Raises:
        ValueError: An argument is not a NumPy ndarray, is not 2-D, has a
            zero-sized axis, contains NaN/inf, or the two shapes differ
            (including cases NumPy could broadcast).

    Example:
        >>> x = np.ones((6, 8), dtype=np.float32)
        >>> h = np.full((6, 8), 0.5, dtype=np.float32)
        >>> add_residual(x, h).shape
        (6, 8)
    """
    _validate_operand(original_input, name="original_input")
    _validate_operand(sublayer_output, name="sublayer_output")

    if original_input.shape != sublayer_output.shape:
        raise ValueError(
            "original_input and sublayer_output must have exactly the same "
            f"shape; got {original_input.shape} and {sublayer_output.shape}. "
            "Broadcasting is not allowed for residual connections."
        )

    # Element-wise sum: each (token, feature) pair is added independently.
    # NumPy "+" allocates a new array; inputs keep their original values.
    return np.add(
        original_input.astype(_DTYPE, copy=False),
        sublayer_output.astype(_DTYPE, copy=False),
    )


def _validate_operand(array: np.ndarray, name: str) -> None:
    """Reject non-arrays, bad ranks/sizes, and non-finite values."""
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"{name} must be a NumPy ndarray, got {type(array).__name__}"
        )
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (sequence_length, embedding_dim), "
            f"got shape {array.shape}"
        )

    sequence_length, embedding_dim = array.shape
    if sequence_length == 0:
        raise ValueError(f"{name} must contain at least one token row")
    if embedding_dim == 0:
        raise ValueError(f"{name} must contain at least one embedding feature")

    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values")
