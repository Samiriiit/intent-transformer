"""Tests for residual (skip) connections.

Run from the project directory:

    pytest tests/test_residual_connection.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from residual_connection import add_residual  # noqa: E402


def test_output_values_equal_elementwise_sum():
    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sublayer = np.array([[0.5, -1.0], [2.0, 0.0]], dtype=np.float32)
    result = add_residual(original, sublayer)
    expected = np.array([[1.5, 1.0], [5.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_output_shape_matches_inputs():
    original = np.ones((6, 8), dtype=np.float32)
    sublayer = np.full((6, 8), 0.25, dtype=np.float32)
    result = add_residual(original, sublayer)
    assert result.shape == (6, 8)
    assert result.shape == original.shape == sublayer.shape


def test_does_not_modify_inputs_in_place():
    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sublayer = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    original_copy = original.copy()
    sublayer_copy = sublayer.copy()
    _ = add_residual(original, sublayer)
    np.testing.assert_array_equal(original, original_copy)
    np.testing.assert_array_equal(sublayer, sublayer_copy)


def test_returns_new_array_object():
    original = np.ones((2, 3), dtype=np.float32)
    sublayer = np.ones((2, 3), dtype=np.float32)
    result = add_residual(original, sublayer)
    assert result is not original
    assert result is not sublayer


def test_adding_zeros_preserves_original_values():
    original = np.array([[1.0, -2.0, 3.5]], dtype=np.float32)
    zeros = np.zeros_like(original)
    result = add_residual(original, zeros)
    np.testing.assert_array_equal(result, original)


def test_rejects_non_numpy_original_input():
    sublayer = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="original_input must be a NumPy"):
        add_residual([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], sublayer)


def test_rejects_non_numpy_sublayer_output():
    original = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="sublayer_output must be a NumPy"):
        add_residual(original, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])


def test_rejects_one_dimensional_inputs():
    with pytest.raises(ValueError, match="2-D"):
        add_residual(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_rejects_three_dimensional_inputs():
    original = np.ones((2, 3, 4), dtype=np.float32)
    sublayer = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        add_residual(original, sublayer)


def test_rejects_mismatched_shapes_even_if_broadcastable():
    original = np.ones((6, 8), dtype=np.float32)
    sublayer = np.ones((1, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly the same shape"):
        add_residual(original, sublayer)


def test_rejects_different_sequence_lengths():
    original = np.ones((6, 8), dtype=np.float32)
    sublayer = np.ones((5, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly the same shape"):
        add_residual(original, sublayer)


def test_rejects_different_embedding_dims():
    original = np.ones((6, 8), dtype=np.float32)
    sublayer = np.ones((6, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly the same shape"):
        add_residual(original, sublayer)


def test_rejects_zero_token_rows():
    original = np.ones((0, 8), dtype=np.float32)
    sublayer = np.ones((0, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one token row"):
        add_residual(original, sublayer)


def test_rejects_zero_embedding_features():
    original = np.ones((4, 0), dtype=np.float32)
    sublayer = np.ones((4, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one embedding feature"):
        add_residual(original, sublayer)


def test_rejects_nan_in_original_input():
    original = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    sublayer = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        add_residual(original, sublayer)


def test_rejects_nan_in_sublayer_output():
    original = np.ones((2, 2), dtype=np.float32)
    sublayer = np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        add_residual(original, sublayer)


def test_rejects_infinite_values():
    original = np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    sublayer = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        add_residual(original, sublayer)

    original_ok = np.ones((2, 2), dtype=np.float32)
    sublayer_neg_inf = np.array([[1.0, -np.inf], [1.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        add_residual(original_ok, sublayer_neg_inf)


def test_does_not_apply_normalization_or_scaling():
    # Pure addition: doubling via residual would require sublayer == original,
    # not a hidden multiply inside add_residual.
    original = np.array([[2.0, 4.0]], dtype=np.float32)
    sublayer = np.array([[3.0, 5.0]], dtype=np.float32)
    result = add_residual(original, sublayer)
    np.testing.assert_array_equal(result, np.array([[5.0, 9.0]], dtype=np.float32))
