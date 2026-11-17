"""Tests for Layer Normalization.

Run from the project directory:

    pytest tests/test_layer_normalization.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from layer_normalization import LayerNormalization  # noqa: E402


def test_gamma_shape():
    ln = LayerNormalization(embedding_dim=8)
    assert ln.gamma.shape == (8,)


def test_beta_shape():
    ln = LayerNormalization(embedding_dim=8)
    assert ln.beta.shape == (8,)


def test_gamma_initialized_to_ones():
    ln = LayerNormalization(embedding_dim=5)
    np.testing.assert_array_equal(ln.gamma, np.ones(5, dtype=np.float32))


def test_beta_initialized_to_zeros():
    ln = LayerNormalization(embedding_dim=5)
    np.testing.assert_array_equal(ln.beta, np.zeros(5, dtype=np.float32))


def test_row_wise_mean_output_shape():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.ones((3, 4), dtype=np.float32)
    mean = ln.calculate_mean(inputs)
    assert mean.shape == (3, 1)


def test_row_wise_variance_output_shape():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.arange(12, dtype=np.float32).reshape(3, 4)
    variance = ln.calculate_variance(inputs)
    assert variance.shape == (3, 1)


def test_forward_output_shape_equals_input_shape():
    ln = LayerNormalization(embedding_dim=8)
    inputs = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    output = ln.forward(inputs)
    assert output.shape == inputs.shape


def test_normalized_rows_have_mean_approximately_zero():
    ln = LayerNormalization(embedding_dim=8)
    inputs = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
    _, _, _, normalized = ln.forward(inputs, return_statistics=True)
    row_means = normalized.mean(axis=-1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-6)


def test_normalized_rows_have_variance_approximately_one_for_non_constant_rows():
    ln = LayerNormalization(embedding_dim=8, epsilon=1e-5)
    inputs = np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32)
    _, _, _, normalized = ln.forward(inputs, return_statistics=True)
    # Population variance across features should be ~1 (epsilon is tiny).
    row_vars = normalized.var(axis=-1, ddof=0)
    np.testing.assert_allclose(row_vars, 1.0, atol=1e-4)


def test_constant_valued_rows_produce_finite_outputs():
    ln = LayerNormalization(embedding_dim=4, epsilon=1e-5)
    inputs = np.full((3, 4), 7.0, dtype=np.float32)
    output = ln.forward(inputs)
    assert np.isfinite(output).all()
    # Constant row -> normalized zeros (with default gamma/beta).
    np.testing.assert_allclose(output, 0.0, atol=1e-6)


def test_epsilon_prevents_division_by_zero():
    ln = LayerNormalization(embedding_dim=3, epsilon=1e-5)
    inputs = np.ones((2, 3), dtype=np.float32)
    mean = ln.calculate_mean(inputs)
    variance = ln.calculate_variance(inputs, mean=mean)
    np.testing.assert_allclose(variance, 0.0, atol=1e-7)
    normalized = ln.normalize(inputs, mean, variance)
    assert np.isfinite(normalized).all()


def test_custom_gamma_changes_the_scale():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.random.default_rng(3).normal(size=(2, 4)).astype(np.float32)
    baseline = ln.forward(inputs)
    ln.gamma = np.full((4,), 2.0, dtype=np.float32)
    scaled = ln.forward(inputs)
    np.testing.assert_allclose(scaled, 2.0 * baseline, rtol=1e-5)


def test_custom_beta_changes_the_offset():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.random.default_rng(4).normal(size=(2, 4)).astype(np.float32)
    baseline = ln.forward(inputs)
    ln.beta = np.full((4,), 0.5, dtype=np.float32)
    shifted = ln.forward(inputs)
    np.testing.assert_allclose(shifted, baseline + 0.5, rtol=1e-5)


def test_deterministic_output_for_repeated_calls():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 2.0, 0.5]], dtype=np.float32)
    first = ln.forward(inputs)
    second = ln.forward(inputs)
    np.testing.assert_array_equal(first, second)


def test_return_statistics_behavior():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    only = ln.forward(inputs, return_statistics=False)
    assert isinstance(only, np.ndarray)
    assert only.shape == (1, 4)

    output, mean, variance, normalized = ln.forward(
        inputs, return_statistics=True
    )
    assert output.shape == (1, 4)
    assert mean.shape == (1, 1)
    assert variance.shape == (1, 1)
    assert normalized.shape == (1, 4)
    np.testing.assert_array_equal(output, ln.apply_scale_and_shift(normalized))


def test_row_independence():
    ln = LayerNormalization(embedding_dim=4)
    inputs = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, -1.0, 0.5, 2.0],
            [4.0, 4.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    baseline = ln.forward(inputs)

    modified = inputs.copy()
    modified[1] = np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    after = ln.forward(modified)

    np.testing.assert_allclose(after[0], baseline[0], rtol=1e-5)
    np.testing.assert_allclose(after[2], baseline[2], rtol=1e-5)
    assert not np.allclose(after[1], baseline[1])


def test_rejection_of_one_dimensional_input():
    ln = LayerNormalization(embedding_dim=4)
    with pytest.raises(ValueError, match="2-D"):
        ln.forward(np.ones(4, dtype=np.float32))


def test_rejection_of_three_dimensional_input():
    ln = LayerNormalization(embedding_dim=4)
    with pytest.raises(ValueError, match="2-D"):
        ln.forward(np.ones((2, 3, 4), dtype=np.float32))


def test_rejection_of_incorrect_embedding_dimension():
    ln = LayerNormalization(embedding_dim=4)
    with pytest.raises(ValueError, match="embedding dimension"):
        ln.forward(np.ones((3, 5), dtype=np.float32))


def test_rejection_of_an_empty_sequence():
    ln = LayerNormalization(embedding_dim=4)
    with pytest.raises(ValueError, match="at least one token"):
        ln.forward(np.ones((0, 4), dtype=np.float32))


def test_rejection_of_nan_input():
    ln = LayerNormalization(embedding_dim=3)
    inputs = np.array([[1.0, np.nan, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        ln.forward(inputs)


def test_rejection_of_infinite_input():
    ln = LayerNormalization(embedding_dim=3)
    inputs = np.array([[1.0, np.inf, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        ln.forward(inputs)
    inputs_neg = np.array([[1.0, -np.inf, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        ln.forward(inputs_neg)


def test_rejection_of_invalid_embedding_dim():
    with pytest.raises(ValueError, match="embedding_dim"):
        LayerNormalization(embedding_dim=0)
    with pytest.raises(ValueError, match="embedding_dim"):
        LayerNormalization(embedding_dim=-3)


def test_rejection_of_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        LayerNormalization(embedding_dim=4, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        LayerNormalization(embedding_dim=4, epsilon=-1e-5)


def test_mean_is_per_row_not_global():
    ln = LayerNormalization(embedding_dim=2)
    inputs = np.array([[0.0, 2.0], [10.0, 30.0]], dtype=np.float32)
    mean = ln.calculate_mean(inputs)
    np.testing.assert_allclose(mean, [[1.0], [20.0]], rtol=1e-5)
