"""Tests for the Post-LN Transformer encoder block.

Run from the project directory:

    pytest tests/test_transformer_encoder_block.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from transformer_encoder_block import TransformerEncoderBlock  # noqa: E402


REQUIRED_KEYS = {
    "attention_output",
    "attention_residual",
    "attention_normalized",
    "feed_forward_output",
    "feed_forward_residual",
    "encoder_output",
}


def _block(seed=0):
    return TransformerEncoderBlock(
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        seed=seed,
    )


def _inputs(seed=0, rows=6, cols=8):
    return np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)


def test_output_shape_equals_input_shape():
    block = _block()
    inputs = _inputs()
    output = block.forward(inputs)
    assert output.shape == inputs.shape == (6, 8)


def test_intermediate_shapes_through_block():
    block = _block()
    inputs = _inputs()
    parts = block.forward(inputs, return_intermediates=True)

    assert parts["attention_output"].shape == (6, 8)
    assert parts["attention_residual"].shape == (6, 8)
    assert parts["attention_normalized"].shape == (6, 8)
    assert parts["feed_forward_output"].shape == (6, 8)
    assert parts["feed_forward_residual"].shape == (6, 8)
    assert parts["encoder_output"].shape == (6, 8)


def test_deterministic_initialization_with_same_seed():
    a = _block(seed=7)
    b = _block(seed=7)
    inputs = _inputs(seed=3)
    np.testing.assert_allclose(a.forward(inputs), b.forward(inputs), rtol=1e-5)


def test_different_seeds_produce_different_initialized_outputs():
    a = _block(seed=1)
    b = _block(seed=2)
    inputs = _inputs(seed=3)
    assert not np.allclose(a.forward(inputs), b.forward(inputs))


def test_repeated_forward_calls_are_identical():
    block = _block(seed=4)
    inputs = _inputs(seed=5)
    first = block.forward(inputs)
    second = block.forward(inputs)
    np.testing.assert_array_equal(first, second)


def test_attention_mask_is_passed_correctly():
    block = _block(seed=8)
    inputs = _inputs(seed=9)
    mask = [1, 1, 1, 1, 0, 0]
    _, weights = block.multi_head_attention.forward(
        inputs, attention_mask=mask, return_attention=True
    )
    # Encoder uses the same MHA path; PAD keys must get ~0 weight.
    output = block.forward(inputs, attention_mask=mask)
    assert output.shape == (6, 8)
    np.testing.assert_allclose(weights[:, :, 4], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights[:, :, 5], 0.0, atol=1e-6)


def test_operation_without_an_attention_mask():
    block = _block(seed=10)
    inputs = _inputs(seed=11)
    output = block.forward(inputs, attention_mask=None)
    assert output.shape == (6, 8)
    assert np.isfinite(output).all()


def test_return_intermediates_false_returns_only_array():
    block = _block()
    result = block.forward(_inputs(), return_intermediates=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 8)


def test_return_intermediates_true_returns_every_required_key():
    block = _block()
    parts = block.forward(_inputs(), return_intermediates=True)
    assert isinstance(parts, dict)
    assert set(parts.keys()) == REQUIRED_KEYS


def test_first_residual_equals_inputs_plus_attention_output():
    block = _block(seed=12)
    inputs = _inputs(seed=13)
    parts = block.forward(inputs, return_intermediates=True)
    expected = inputs + parts["attention_output"]
    np.testing.assert_allclose(parts["attention_residual"], expected, rtol=1e-5)


def test_second_residual_equals_attention_normalized_plus_ffn_output():
    block = _block(seed=14)
    inputs = _inputs(seed=15)
    parts = block.forward(inputs, return_intermediates=True)
    expected = parts["attention_normalized"] + parts["feed_forward_output"]
    np.testing.assert_allclose(
        parts["feed_forward_residual"], expected, rtol=1e-5
    )


def test_layer_norm_instances_are_separate():
    block = _block()
    assert block.layer_norm_1 is not block.layer_norm_2
    # Distinct parameter objects as well.
    assert block.layer_norm_1.gamma is not block.layer_norm_2.gamma
    assert block.layer_norm_1.beta is not block.layer_norm_2.beta


def test_embedding_dim_not_divisible_by_num_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        TransformerEncoderBlock(
            embedding_dim=10, num_heads=3, hidden_dim=16, seed=0
        )


def test_invalid_one_dimensional_input():
    block = _block()
    with pytest.raises(ValueError, match="2-D"):
        block.forward(np.ones(8, dtype=np.float32))


def test_invalid_three_dimensional_input():
    block = _block()
    with pytest.raises(ValueError, match="2-D"):
        block.forward(np.ones((2, 3, 8), dtype=np.float32))


def test_incorrect_embedding_dimension():
    block = _block()
    with pytest.raises(ValueError, match="embedding dimension"):
        block.forward(np.ones((4, 4), dtype=np.float32))


def test_empty_sequence():
    block = _block()
    with pytest.raises(ValueError, match="at least one token"):
        block.forward(np.ones((0, 8), dtype=np.float32))


def test_nan_input():
    block = _block()
    inputs = _inputs()
    inputs[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        block.forward(inputs)


def test_infinite_input():
    block = _block()
    inputs = _inputs()
    inputs[1, 2] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        block.forward(inputs)


def test_invalid_attention_mask_length():
    block = _block()
    with pytest.raises(ValueError, match="length"):
        block.forward(_inputs(), attention_mask=[1, 1, 1])


def test_invalid_attention_mask_values():
    block = _block()
    with pytest.raises(ValueError, match="0 and 1"):
        block.forward(_inputs(), attention_mask=[1, 1, 1, 2, 0, 0])


def test_fully_masked_sequence_rejection():
    block = _block()
    with pytest.raises(ValueError, match="completely masked"):
        block.forward(_inputs(), attention_mask=[0, 0, 0, 0, 0, 0])


def test_input_remains_unchanged_after_forward():
    block = _block(seed=20)
    inputs = _inputs(seed=21)
    snapshot = inputs.copy()
    _ = block.forward(inputs, attention_mask=[1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(inputs, snapshot)


def test_encoder_output_contains_only_finite_values():
    block = _block(seed=22)
    output = block.forward(_inputs(seed=23))
    assert np.isfinite(output).all()


def test_non_numpy_input_rejected():
    block = _block()
    with pytest.raises(ValueError, match="NumPy ndarray"):
        block.forward([[0.0] * 8] * 3)


def test_attention_and_ffn_use_distinct_seeds():
    block = TransformerEncoderBlock(
        embedding_dim=8, num_heads=2, hidden_dim=16, seed=100
    )
    # Documented rule: attention gets seed, FFN gets seed+1 → different weights.
    assert not np.array_equal(
        block.multi_head_attention.W_query[:2, :2],
        block.feed_forward.W1[:2, :2],
    )
