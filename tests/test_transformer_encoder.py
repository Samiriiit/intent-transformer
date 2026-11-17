"""Tests for the stacked Transformer encoder.

Run from the project directory:

    pytest tests/test_transformer_encoder.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from transformer_encoder import TransformerEncoder  # noqa: E402
from transformer_encoder_block import TransformerEncoderBlock  # noqa: E402


def _encoder(num_layers=2, seed=0):
    return TransformerEncoder(
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        num_layers=num_layers,
        seed=seed,
    )


def _inputs(seed=0, rows=6, cols=8):
    return np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)


def test_exactly_num_layers_blocks_are_created():
    encoder = _encoder(num_layers=3, seed=1)
    assert len(encoder.layers) == 3
    assert all(isinstance(layer, TransformerEncoderBlock) for layer in encoder.layers)


def test_every_block_is_a_distinct_object():
    encoder = _encoder(num_layers=3, seed=2)
    assert encoder.layers[0] is not encoder.layers[1]
    assert encoder.layers[1] is not encoder.layers[2]
    assert encoder.layers[0] is not encoder.layers[2]


def test_attention_components_are_distinct_across_layers():
    encoder = _encoder(num_layers=2, seed=3)
    assert (
        encoder.layers[0].multi_head_attention
        is not encoder.layers[1].multi_head_attention
    )


def test_feed_forward_components_are_distinct_across_layers():
    encoder = _encoder(num_layers=2, seed=4)
    assert encoder.layers[0].feed_forward is not encoder.layers[1].feed_forward


def test_normalization_components_are_distinct_across_layers():
    encoder = _encoder(num_layers=2, seed=5)
    assert encoder.layers[0].layer_norm_1 is not encoder.layers[1].layer_norm_1
    assert encoder.layers[0].layer_norm_2 is not encoder.layers[1].layer_norm_2


def test_weight_arrays_do_not_share_memory_across_layers():
    encoder = _encoder(num_layers=2, seed=6)
    left = encoder.layers[0]
    right = encoder.layers[1]

    pairs = [
        (left.multi_head_attention.W_query, right.multi_head_attention.W_query),
        (left.multi_head_attention.W_key, right.multi_head_attention.W_key),
        (left.multi_head_attention.W_value, right.multi_head_attention.W_value),
        (left.multi_head_attention.W_output, right.multi_head_attention.W_output),
        (left.feed_forward.W1, right.feed_forward.W1),
        (left.feed_forward.W2, right.feed_forward.W2),
        (left.layer_norm_1.gamma, right.layer_norm_1.gamma),
        (left.layer_norm_1.beta, right.layer_norm_1.beta),
        (left.layer_norm_2.gamma, right.layer_norm_2.gamma),
        (left.layer_norm_2.beta, right.layer_norm_2.beta),
    ]
    for a, b in pairs:
        assert a is not b
        assert not np.shares_memory(a, b)


def test_output_shape_equals_input_shape():
    encoder = _encoder(num_layers=2, seed=7)
    inputs = _inputs(seed=8)
    output = encoder.forward(inputs)
    assert output.shape == inputs.shape == (6, 8)


def test_output_after_every_layer_has_expected_shape():
    encoder = _encoder(num_layers=3, seed=9)
    result = encoder.forward(_inputs(seed=10), return_all_layers=True)
    assert all(layer_out.shape == (6, 8) for layer_out in result["layer_outputs"])


def test_final_output_equals_last_entry_in_layer_outputs():
    encoder = _encoder(num_layers=3, seed=11)
    result = encoder.forward(_inputs(seed=12), return_all_layers=True)
    np.testing.assert_array_equal(
        result["final_output"], result["layer_outputs"][-1]
    )


def test_layer_outputs_length_equals_num_layers():
    encoder = _encoder(num_layers=4, seed=13)
    result = encoder.forward(_inputs(seed=14), return_all_layers=True)
    assert len(result["layer_outputs"]) == 4


def test_output_of_layer_1_is_used_as_input_to_layer_2():
    encoder = _encoder(num_layers=2, seed=15)
    inputs = _inputs(seed=16)

    # Manual two-step run with the same layer objects.
    after_first = encoder.layers[0].forward(inputs)
    after_second = encoder.layers[1].forward(after_first)

    result = encoder.forward(inputs, return_all_layers=True)
    np.testing.assert_allclose(result["layer_outputs"][0], after_first, rtol=1e-5)
    np.testing.assert_allclose(result["layer_outputs"][1], after_second, rtol=1e-5)
    np.testing.assert_allclose(result["final_output"], after_second, rtol=1e-5)


def test_layer_outputs_are_independent_copies():
    encoder = _encoder(num_layers=2, seed=17)
    result = encoder.forward(_inputs(seed=18), return_all_layers=True)
    first = result["layer_outputs"][0]
    first_snapshot = first.copy()
    first[:] = 0.0
    # Mutating the returned list entry must not change the final output copy.
    assert not np.array_equal(result["final_output"], 0)
    np.testing.assert_array_equal(
        encoder.forward(_inputs(seed=18), return_all_layers=True)["layer_outputs"][0],
        first_snapshot,
    )


def test_same_attention_mask_accepted_for_every_layer():
    encoder = _encoder(num_layers=2, seed=19)
    inputs = _inputs(seed=20)
    mask = [1, 1, 1, 1, 0, 0]
    output = encoder.forward(inputs, attention_mask=mask)
    assert output.shape == (6, 8)
    assert np.isfinite(output).all()


def test_return_all_layers_false_returns_only_array():
    encoder = _encoder(num_layers=2, seed=21)
    result = encoder.forward(_inputs(seed=22), return_all_layers=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 8)


def test_deterministic_with_same_seed():
    a = _encoder(num_layers=2, seed=30)
    b = _encoder(num_layers=2, seed=30)
    inputs = _inputs(seed=31)
    np.testing.assert_allclose(a.forward(inputs), b.forward(inputs), rtol=1e-5)


def test_different_seeds_produce_different_outputs():
    a = _encoder(num_layers=2, seed=40)
    b = _encoder(num_layers=2, seed=41)
    inputs = _inputs(seed=42)
    assert not np.allclose(a.forward(inputs), b.forward(inputs))


def test_layer_seeds_follow_seed_plus_100_times_index():
    encoder = TransformerEncoder(
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=3,
        seed=1000,
    )
    # Rebuild expected blocks with documented seeds and compare one weight.
    expected_seeds = [1000, 1100, 1200]
    for layer, expected_seed in zip(encoder.layers, expected_seeds):
        reference = TransformerEncoderBlock(
            embedding_dim=8, num_heads=2, hidden_dim=16, seed=expected_seed
        )
        np.testing.assert_array_equal(
            layer.multi_head_attention.W_query,
            reference.multi_head_attention.W_query,
        )


def test_inputs_unchanged_after_forward():
    encoder = _encoder(num_layers=2, seed=50)
    inputs = _inputs(seed=51)
    snapshot = inputs.copy()
    _ = encoder.forward(inputs, attention_mask=[1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(inputs, snapshot)


def test_invalid_constructor_dimensions():
    with pytest.raises(ValueError, match="num_layers"):
        TransformerEncoder(
            embedding_dim=8, num_heads=2, hidden_dim=16, num_layers=0
        )
    with pytest.raises(ValueError, match="divisible"):
        TransformerEncoder(
            embedding_dim=10, num_heads=3, hidden_dim=16, num_layers=2
        )
    with pytest.raises(ValueError, match="epsilon"):
        TransformerEncoder(
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            epsilon=0.0,
        )


def test_invalid_one_dimensional_input():
    encoder = _encoder()
    with pytest.raises(ValueError, match="2-D"):
        encoder.forward(np.ones(8, dtype=np.float32))


def test_incorrect_embedding_dimension():
    encoder = _encoder()
    with pytest.raises(ValueError, match="embedding dimension"):
        encoder.forward(np.ones((4, 4), dtype=np.float32))


def test_empty_sequence():
    encoder = _encoder()
    with pytest.raises(ValueError, match="at least one token"):
        encoder.forward(np.ones((0, 8), dtype=np.float32))


def test_nan_and_inf_rejected():
    encoder = _encoder()
    nan_inputs = _inputs()
    nan_inputs[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        encoder.forward(nan_inputs)

    inf_inputs = _inputs(seed=99)
    inf_inputs[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        encoder.forward(inf_inputs)


def test_invalid_attention_mask_cases():
    encoder = _encoder()
    inputs = _inputs()
    with pytest.raises(ValueError, match="length"):
        encoder.forward(inputs, attention_mask=[1, 1])
    with pytest.raises(ValueError, match="0 and 1"):
        encoder.forward(inputs, attention_mask=[1, 1, 1, 2, 0, 0])
    with pytest.raises(ValueError, match="completely masked"):
        encoder.forward(inputs, attention_mask=[0, 0, 0, 0, 0, 0])


def test_non_numpy_input_rejected():
    encoder = _encoder()
    with pytest.raises(ValueError, match="NumPy ndarray"):
        encoder.forward([[0.0] * 8] * 3)


def test_final_output_is_finite():
    encoder = _encoder(num_layers=3, seed=60)
    output = encoder.forward(_inputs(seed=61))
    assert np.isfinite(output).all()
