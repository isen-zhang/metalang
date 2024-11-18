from seqthetic.seed import get_rngs
from seqthetic.domains.base import BaseDomainSpec


import numpy as np


from typing import List

from seqthetic.range import FlexibleRange


class NGram(BaseDomainSpec):
    type: str = "ngram"
    num_vocab: int
    sequence_length: FlexibleRange
    gram_count: int = 2

    def make_sequences(self, num_token: int, seed: int) -> List[np.ndarray]:
        np.random.seed(seed)
        sequences = []

        # Generate a single probability matrix for all sequences
        prob = np.random.random((self.num_vocab**self.gram_count, self.num_vocab))
        prob = prob / prob.sum(axis=1)[:, None]
        print(prob)
        if self.sequence_length.constant:
            num_sequence = num_token // self.sequence_length.min
        else:
            # For variable sequence length, we'll use the average length
            avg_length = (self.sequence_length.min + self.sequence_length.max) / 2
            num_sequence = int(num_token / avg_length)

        vocab_sample_rngs = get_rngs(seed, [("vocab", num_sequence)])

        for seq_idx in range(num_sequence):
            rng = vocab_sample_rngs[seq_idx]

            # Determine the sequence length
            if self.sequence_length.constant:
                seq_length = self.sequence_length.min
            else:
                seq_length = rng.integers(
                    self.sequence_length.min, self.sequence_length.max + 1
                )

            # Initialize the sequence with random first n-gram
            sequence = rng.choice(self.num_vocab, self.gram_count).tolist()

            # Generate the rest of the sequence
            while len(sequence) < seq_length:
                context = tuple(sequence[-(self.gram_count - 1) :])
                context_index = self._context_to_index(context)
                next_word = rng.choice(self.num_vocab, p=prob[context_index])
                sequence.append(next_word)

            sequences.append(np.array(sequence))

        return sequences

    def _context_to_index(self, context: tuple) -> int:
        """Convert a context (n-gram) to an index in the probability matrix."""
        index = 0
        for i, word in enumerate(context):
            index += word * (self.num_vocab**i)
        return index
