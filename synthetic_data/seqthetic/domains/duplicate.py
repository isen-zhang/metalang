from pydantic import Field
from tqdm import tqdm
from seqthetic.vocabulary import Vocabulary
from seqthetic.seed import get_rngs, make_seed, spawn_rng_list
from seqthetic.domains.base import BaseDomainSpec
from seqthetic.range import FlexibleRange, Range
from seqthetic.utils import (
    flatten,
    get_seqlen_and_numseq,
    sample_integer_range,
    DelimiterTokenField,
)


import numpy as np


class Duplicate(BaseDomainSpec):
    """
    Duplicates each token for `num_duplicate` times

    Demo:
    input: c a b
    output: c c a a b b
    sample:c a b <sep> c c a a b b

    Params:
    num_duplicate: how many times one token is repeated

    Usage:

    domain = Duplicate(
        sequence_length=1024,
        num_duplicate=2,
    )
    """
    type: str = "duplicate"
    sequence_length: FlexibleRange
    delimiter: bool = True
    vocab: Vocabulary | None
    delimiter_tokens: list[int] = DelimiterTokenField
    num_duplicate: FlexibleRange = Range(min=2, max=2)

    def make_sequences(
        self,
        num_token: int,
        seed: int | None = None,
        num_sequence: int = 1,
    ):
        if not self.vocab:
            raise ValueError("Vocabulary is required for Duplicate domain")
        seed = seed or make_seed()
        (vocab_sample_rng, num_duplicate_rng, seqlen_rng) = get_rngs(
            seed, ["vocab", "num_duplicate", "sequence_length"]
        )
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, seqlen_rng
        )
        vocab_sample_rng = spawn_rng_list(vocab_sample_rng, num_sequence)

        sequences = []

        delimiter = self.delimiter_tokens if self.delimiter else []
        num_duplicates = sample_integer_range(
            self.num_duplicate, num_duplicate_rng, num_sequence
        )
        for rng, num_duplicate, sequence_length in tqdm(
            zip(vocab_sample_rng, num_duplicates, sequence_lengths)
        ):
            unique_tokens = int((sequence_length - 1) / (num_duplicate + 1))
            sampled = self.vocab.sample_vocab(unique_tokens, rng)
            duplicated = np.array(
                flatten([[token] * num_duplicate for token in sampled])
            )

            sequences.append(np.concatenate([sampled, delimiter, duplicated]))

        return sequences


if __name__ == "__main__":
    domain = Duplicate(
        sequence_length=10,
        num_duplicate=2,
        vocab=Vocabulary(num_vocab=100),
    )
    sequences = domain.make_sequences(10, seed=0)
    print(sequences)
