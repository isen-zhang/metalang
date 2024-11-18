from seqthetic.utils import get_seqlen_and_numseq, DelimiterTokenField
from seqthetic.vocabulary import Vocabulary
from seqthetic.seed import get_rngs, make_seed, spawn_rng_list
from seqthetic.domains.base import BaseDomainSpec


import numpy as np

from seqthetic.range import FlexibleRange


class Identity(BaseDomainSpec):
    '''
    Given a sequence, the task copies the sequence (or in reverse).
    
    
    Input: a b c
    Output: a b c
    
    '''
    type: str = "identity"
    sequence_length: FlexibleRange
    reverse: bool = False
    delimiter: bool = True
    delimiter_tokens: list[int] = DelimiterTokenField

    vocab: Vocabulary

    def make_sequences(self, num_token: int, seed: int | None = None):
        seed = seed or make_seed()
        sequences = []
        (vocab_sample_rng, sequence_length) = get_rngs(
            seed, ["vocab", "sequence_length"]
        )
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, sequence_length
        )
        vocab_sample_rng = spawn_rng_list(vocab_sample_rng, num_sequence)
        delimiter = self.delimiter_tokens if self.delimiter else []

        for rng, seqlen in zip(vocab_sample_rng, sequence_lengths):
            sampled = self.vocab.sample_vocab(seqlen, rng)
            output = sampled if not self.reverse else sampled[::-1]

            sequences.append(np.concatenate([sampled, delimiter, output]))

        return sequences


if __name__ == "__main__":

    domain = Identity(
        sequence_length=10,
        vocab=Vocabulary(num_vocab=20),
        reverse=True,
        delimiter=True,
    )
    sequences = domain.make_sequences(10, make_seed())
    print(sequences)

    domain = Identity(
        sequence_length=10,
        vocab=Vocabulary(num_vocab=20),
        reverse=False,
        delimiter=True,
    )
    sequences = domain.make_sequences(10, make_seed())
    print(sequences)
