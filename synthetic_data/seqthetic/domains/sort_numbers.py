from seqthetic.seed import get_rngs, make_seed, spawn_rng_list
from seqthetic.domains.base import BaseDomainSpec
from seqthetic.range import FlexibleRange, Range


import numpy as np

from seqthetic.utils import get_seqlen_and_numseq


class SortNumbers(BaseDomainSpec):
    type: str = "sort_numbers"
    sequence_length: FlexibleRange
    nums: Range
    delimiter: bool = True

    def make_sequences(self, num_token: int, seed: int | None = None):
        seed = seed or make_seed()
        sequences = []
        (sample_rngs, seqlen_rng) = get_rngs(seed, ["sample_rngs", "sequence_length"])
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, seqlen_rng
        )
        sample_rngs = spawn_rng_list(sample_rngs, num_sequence)

        nums = np.arange(self.nums.min, self.nums.max)
        delimiter = [-1] if self.delimiter else []
        for rng, sequence_length in zip(sample_rngs, sequence_lengths):
            sampled = rng.choice(nums, int(sequence_length), replace=False)
            sequences.append(np.concatenate([sampled, delimiter, np.sort(sampled)]))

        return sequences

if __name__ == '__main__':
    domain = SortNumbers(
        sequence_length=10,
        nums=Range(min=1, max=40),
        delimiter=True,
        
    )
    sequences = domain.make_sequences(10, make_seed())
    print(sequences)