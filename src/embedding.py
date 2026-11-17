"""Token embedding lookup for the intent-classification transformer.

This module turns integer token ids into dense vectors. It does not implement
positional encoding, attention, or any transformer block. Those layers will
consume the vectors this class returns.

Why token ids are not enough
----------------------------
A token id is a category label: "nike" might be 3 and "shoes" might be 4.
Those integers have no geometric meaning. The model cannot add them, compare
distances, or learn that "nike" is closer to "adidas" than to "lease". A
neural network needs real-valued vectors so gradient descent can move each
word a little in a continuous space.

Why embeddings are needed
-------------------------
An embedding matrix has one row per vocabulary entry. Looking up id 3 copies
row 3 into the sequence. After the lookup, a query such as [2, 3, 4, 5]
becomes a matrix of shape (sequence_length, embedding_dim). Later layers
(attention, feed-forward) operate on that matrix.

How embeddings are learned during training
------------------------------------------
At first the rows are random numbers. During training the model predicts an
intent, compares it to the true label, and backpropagates an error. The
gradient for each used row is applied with a small learning rate, so words
that often appear in ECOMMERCE queries drift toward each other, and words
that appear in LEGAL queries drift another way. This file only stores and
looks up the matrix. Updating the rows is left to a future training loop.

Pipeline position
-----------------
tokenizer.encode -> dataset padding -> TokenEmbedding.forward -> (later) model
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

# float32 is enough for a teaching model and matches what PyTorch will use later.
_DTYPE = np.float32

TokenIds = Union[Sequence[int], np.ndarray]


class TokenEmbedding:
    """A vocab_size by embedding_dim table of learnable token vectors.

    The class exists because the embedding matrix is mutable state. Training
    will later write new values into ``self.weight``. A plain function would
    have nowhere durable to keep that matrix between batches.

    Attributes:
        vocab_size: Number of rows (one per token id, including <PAD> and <UNK>).
        embedding_dim: Number of columns (length of each vector).
        weight: The matrix itself, shape (vocab_size, embedding_dim).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seed: int | None = None,
        scale: float = 0.02,
    ) -> None:
        """Create a randomly initialized embedding matrix.

        Small random values (controlled by ``scale``) keep early activations
        from exploding before training has started. A fixed ``seed`` makes
        tests and demos reproducible.

        Args:
            vocab_size: Must match the tokenizer vocabulary size.
            embedding_dim: Length of each token vector. Common teaching sizes
                are 32, 64, or 128.
            seed: Optional RNG seed. ``None`` draws a fresh random matrix.
            scale: Standard deviation for the normal initializer.

        Raises:
            ValueError: vocab_size or embedding_dim is less than 1.
        """
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)
        # Shape: one vector per token id. Row i is the embedding for token id i.
        self.weight: np.ndarray = (
            rng.normal(loc=0.0, scale=scale, size=(vocab_size, embedding_dim)).astype(
                _DTYPE, copy=False
            )
        )

    def forward(self, input_ids: TokenIds) -> np.ndarray:
        """Look up an embedding vector for every id in a sequence.

        This is the same operation a future ``nn.Embedding`` layer will do:
        advanced indexing into the weight matrix. No matrix multiply happens
        here; each id simply selects a row.

        Args:
            input_ids: Token ids for one sequence, for example ``[2, 3, 4, 5]``.
                May be a Python list or a 1-D NumPy array.

        Returns:
            Array of shape ``(sequence_length, embedding_dim)``. An empty
            input returns shape ``(0, embedding_dim)``.

        Raises:
            ValueError: ``input_ids`` is not 1-D, or an id is out of range.

        Example:
            ids ``[2, 3, 4, 5]`` -> four rows stacked, each of length
            ``embedding_dim``.
        """
        ids = self._as_id_array(input_ids)
        self._validate_ids(ids)
        # NumPy row indexing: weight[[2,3,4,5]] -> shape (4, embedding_dim).
        return self.weight[ids]

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Return the vector for a single token id.

        Useful when you want to inspect one row (for example the <PAD> vector)
        without wrapping it in a sequence.

        Args:
            token_id: Integer id from the vocabulary.

        Returns:
            A 1-D array of shape ``(embedding_dim,)``.

        Raises:
            ValueError: ``token_id`` is outside ``[0, vocab_size)``.
        """
        if not isinstance(token_id, (int, np.integer)):
            raise ValueError(f"token_id must be an int, got {type(token_id)!r}")
        index = int(token_id)
        if index < 0 or index >= self.vocab_size:
            raise ValueError(
                f"token_id {index} is outside [0, {self.vocab_size})"
            )
        return self.weight[index]

    def embedding_shape(self) -> Tuple[int, int]:
        """Return ``(vocab_size, embedding_dim)``.

        Handy in tests and when wiring the next layer, which must accept
        vectors of length ``embedding_dim``.
        """
        return (self.vocab_size, self.embedding_dim)

    def _as_id_array(self, input_ids: TokenIds) -> np.ndarray:
        """Convert a sequence of ids to a 1-D int array."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 0:
            # A lone integer becomes a length-1 sequence.
            ids = ids.reshape(1)
        if ids.ndim != 1:
            raise ValueError(
                f"input_ids must be 1-D, got shape {ids.shape}"
            )
        return ids

    def _validate_ids(self, ids: np.ndarray) -> None:
        """Reject ids that cannot index into the weight matrix."""
        if ids.size == 0:
            return
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            bad = ids[(ids < 0) | (ids >= self.vocab_size)]
            raise ValueError(
                f"token id(s) {bad.tolist()} outside [0, {self.vocab_size})"
            )
