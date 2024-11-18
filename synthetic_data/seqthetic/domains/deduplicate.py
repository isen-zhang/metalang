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


class Deduplicate(BaseDomainSpec):
    """
    Function:
    Removes duplicates from the sequence. Only keeps first unique occurrence of each tokens.

    Examples:
    input: c a a b b
    output: c a b
    sample: c a a b b <sep> c a b

    input: c b a a b
    output: c b a


    Params:
    unique_token_ratio: number of unique tokens relative to the sequence_length; smaller than 0.5

    repeat_token_ratio: number of tokens chosen to be repeated relative to unique tokens; smaller than 0.5
    """
    type: str = "deduplicate"
    sequence_length: FlexibleRange
    unique_token_ratio: FlexibleRange
    repeat_token_ratio: FlexibleRange
    repeat_token_times: FlexibleRange = 2
    delimiter: bool = True
    delimiter_tokens: list[int] = DelimiterTokenField

    vocab: Vocabulary

    def make_sequences(self, num_token: int, seed: int | None = None):
        # get seed
        seed = seed or make_seed()
        sequences = []

        (
            sample_rng,
            unique_token_rng,
            repeat_token_rng,
            sequence_length_rng,
        ) = get_rngs(
            seed, ["sample_rng", "unique_token", "repeat_token", "sequence_length"]
        )
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, sequence_length_rng
        )
        sample_rngs = spawn_rng_list(sample_rng, num_sequence)

        # unique token for each sequence
        unique_tokens = unique_token_rng.uniform(
            self.unique_token_ratio.min, self.unique_token_ratio.max, num_sequence
        ) * np.array(sequence_lengths)

        ratios = (
            repeat_token_rng.uniform(
                self.repeat_token_ratio.min, self.repeat_token_ratio.max, num_sequence
            )
            if not self.repeat_token_ratio.constant
            else [self.repeat_token_ratio.min] * num_sequence
        )

        repeat_tokens = [
            int(ratio * unique_token)
            for ratio, unique_token in zip(ratios, unique_tokens)
        ]
        delimiter = self.delimiter_tokens if self.delimiter else []
        for sample, unique_token, repeat_token in tqdm(
            zip(sample_rngs, unique_tokens, repeat_tokens)
        ):
            # sample vocab
            sampled = self.vocab.sample_vocab(unique_token, sample)
            # choose tokens to be repeated
            # 1 means repeated, 0 means not repeated
            repeated = np.zeros(int(unique_token), dtype=int)
            repeated[:repeat_token] = 1
            np.random.shuffle(repeated)
            tokens = []
            for token, should_repeat in zip(sampled, repeated):
                if should_repeat:
                    # 随机决定重复次数，比如2-4次
                    repeat_times = 2
                    tokens.extend([token] * repeat_times)
                else:
                    tokens.append(token)
            sequences.append(np.concatenate([tokens, delimiter, sampled]))

        return sequences


if __name__ == "__main__":
    dedup = Deduplicate(
        sequence_length=20,
        unique_token_ratio=Range(min=0.1, max=0.5),
        repeat_token_ratio=1,
        vocab=Vocabulary(num_vocab=10),
    )

    seqs = dedup.make_sequences(num_token=20, seed=42)
    print(seqs)
