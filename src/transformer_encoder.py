"""Stacked Transformer encoder (NumPy): several independent Post-LN blocks.

This module runs a sequence through ``num_layers`` separate
``TransformerEncoderBlock`` instances in order. It does not classify intents,
pool tokens, compute logits/loss, or train parameters.

Why stack encoder blocks
------------------------
One block mixes tokens once (attention) and transforms them once (FFN). Stacking
lets later blocks refine already-contextual representations: early layers form
local or shallow patterns; deeper layers can combine those into richer features.
Depth is a capacity choice—``num_layers`` controls how many times that refine
step runs. Deeper is not automatically better; without training, extra layers
are just more random transforms.

Independent parameters per layer
--------------------------------
Each list entry is a **new** ``TransformerEncoderBlock`` with its own attention,
feed-forward, and LayerNorm weights. Layers do not share objects or arrays.
Reusing one block repeatedly would force every depth to use the same matrices.

Shape invariance
----------------
Every block maps ``(seq, d) -> (seq, d)``. The stack therefore keeps
``sequence_length`` and ``embedding_dim`` unchanged from input to final output.

Padding mask
------------
The same ``attention_mask`` is passed into **every** block. PAD keys must stay
masked at each depth. The sequence is not shortened; PAD rows are not zeroed
here. Post-LN details live inside each block.

Seed strategy
-------------
If ``seed`` is ``None``, every block receives ``None``.
If ``seed`` is an int, layer ``i`` receives ``seed + 100 * i``
(layer 0 → ``seed``, layer 1 → ``seed + 100``, layer 2 → ``seed + 200``, ...).
That keeps initialization deterministic while giving each depth distinct weights.

Pipeline position
-----------------
TokenEmbedding + PositionalEncoding
    -> TransformerEncoder.forward   (this module)
    -> (later) pooling / classifier
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from transformer_encoder_block import TransformerEncoderBlock

_DTYPE = np.float32


class TransformerEncoder:
    """Depth-``num_layers`` stack of Post-LN encoder blocks.

    Parameters
    ----------
    embedding_dim:
        Token feature width (input and output of every layer).
    num_heads:
        Attention heads per block; must divide ``embedding_dim``.
    hidden_dim:
        Feed-forward hidden width inside each block.
    num_layers:
        Number of independent ``TransformerEncoderBlock`` instances.
    epsilon:
        LayerNorm epsilon forwarded to every block.
    seed:
        Optional base seed. Layer ``i`` uses ``seed + 100 * i``, or ``None``
        for all layers when ``seed is None``.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        epsilon: float = 1e-5,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(embedding_dim, (int, np.integer)) or int(embedding_dim) < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(num_heads, (int, np.integer)) or int(num_heads) < 1:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(hidden_dim, (int, np.integer)) or int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(num_layers, (int, np.integer)) or int(num_layers) < 1:
            raise ValueError("num_layers must be a positive integer")
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
        self.num_layers = int(num_layers)
        self.epsilon = float(epsilon)

        # One fresh block per depth — never append the same object twice.
        self.layers: List[TransformerEncoderBlock] = []
        for layer_index in range(self.num_layers):
            layer_seed = None if seed is None else seed + 100 * layer_index
            self.layers.append(
                TransformerEncoderBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    epsilon=self.epsilon,
                    seed=layer_seed,
                )
            )

    def forward(
        self,
        inputs: np.ndarray,
        attention_mask: Optional[Union[np.ndarray, list]] = None,
        return_all_layers: bool = False,
    ) -> Union[np.ndarray, Dict[str, Union[np.ndarray, List[np.ndarray]]]]:
        """Pass the sequence through every encoder block in order.

        The output of layer ``i`` becomes the input of layer ``i + 1``. The
        original ``inputs`` tensor is only used as the first layer's input.
        The same ``attention_mask`` is forwarded to each block.

        Args:
            inputs: Shape ``(sequence_length, embedding_dim)``.
            attention_mask: Optional length-``sequence_length`` mask of 0/1.
            return_all_layers: When True, also return a copy of each layer's
                output for inspection and tests.

        Returns:
            Final array of shape ``(sequence_length, embedding_dim)``, or::

                {
                    "final_output": <ndarray>,
                    "layer_outputs": [<after layer 0>, <after layer 1>, ...],
                }

            Each entry in ``layer_outputs`` is an independent copy.
        """
        current_output = self._validate_inputs(inputs)
        if attention_mask is not None:
            self._validate_attention_mask(attention_mask, current_output.shape[0])

        layer_outputs: List[np.ndarray] = []

        # Depth loop: refine representations without changing (seq, d).
        for layer in self.layers:
            current_output = layer.forward(
                current_output,
                attention_mask=attention_mask,
            )
            if return_all_layers:
                # Copy so callers cannot mutate tensors still feeding the stack
                # on a future experimental re-run of the same list.
                layer_outputs.append(np.array(current_output, copy=True))

        if return_all_layers:
            return {
                "final_output": np.array(current_output, copy=True),
                "layer_outputs": layer_outputs,
            }
        return current_output

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
