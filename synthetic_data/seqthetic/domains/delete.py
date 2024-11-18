from tqdm import tqdm
from seqthetic.vocabulary import Vocabulary
from seqthetic.seed import get_rngs, make_seed, spawn_rng_list
from seqthetic.domains.base import BaseDomainSpec
from seqthetic.range import FlexibleRange, Range
from seqthetic.utils import (
    get_seqlen_and_numseq,
    sample_integer_range,
    DelimiterTokenField,
)


import numpy as np


class Delete(BaseDomainSpec):
    """
    Function:
    Deletes substring(seq_to_delete) from source sequence and return a new sequence. The substring may not exist in the original string.

    Examples:
    input: c a b c
    seq_to_delete: a b
    output: c c
    sample: c a b c <sep> a b <sep> c c

    input: c a b c
    seq_to_delete: p q
    output: c a b c
    sample: c a b c <sep> p q <sep> c a b c

    Params:
    should_delete_ratio: how many sequences should have a substring deleted. It's a float, do not pass a range.
    seq_to_delete_length: how long should the seq_to_delete be
    """
    type: str = "delete"
    sequence_length: FlexibleRange
    should_delete_ratio: float
    seq_to_delete_length: FlexibleRange
    delimiter: bool = True
    delimiter_tokens: list[int] = DelimiterTokenField

    vocab: Vocabulary

    def make_sequences(self, num_token: int, seed: int | None = None):
        # 是否要删除；原序列的 token 数 (L/2-1)；要删除的 token 数；要删除的 token 在原token中的起始位置
        seed = seed or make_seed()
        sequences = []

        (
            should_delete_rng,
            delete_length_rng,
            delete_index_rng,
            vocab_sample_rng,
            seq_length_rng,
            sequence_choice_rng,
        ) = get_rngs(
            seed,
            [
                "should_delete",
                "delete_length",
                "delete_index",
                "vocab",
                # ("vocab", num_sequence),
                "sequence_length",
                "sequence_choice",
                # ("sequence_choice", num_sequence),
            ],
        )
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, seq_length_rng
        )
        vocab_sample_rngs = spawn_rng_list(vocab_sample_rng, num_sequence)
        sequence_choice_rngs = spawn_rng_list(sequence_choice_rng, num_sequence)

        seqs_should_delete = should_delete_rng.binomial(
            n=1, p=self.should_delete_ratio, size=int(num_sequence)
        )

        delete_lengths = sample_integer_range(
            self.seq_to_delete_length, delete_length_rng, num_sequence
        )
        source_sequence_length = self.sequence_length.min // 2 - 1
        if self.sequence_length.constant:
            delete_indexes = delete_index_rng.integers(
                0, source_sequence_length, num_sequence
            )
        else:
            delete_indexes = sample_integer_range(
                self.sequence_length, delete_index_rng, num_sequence
            )
        delimiter = self.delimiter_tokens if self.delimiter else []
        for (
            should_delete,
            vocab_rng,
            delete_length,
            delete_index,
            sequence_choice_rng,
            sequence_length,
        ) in tqdm(
            zip(
                seqs_should_delete,
                vocab_sample_rngs,
                delete_lengths,
                delete_indexes,
                sequence_choice_rngs,
                sequence_lengths,
            )
        ):

            if not should_delete:
                sampled = self.vocab.sample_vocab(
                    (sequence_length + delete_length) / 2 - 1, vocab_rng
                )
                to_delete = sequence_choice_rng.choice(
                    sampled, delete_length, replace=False
                )
                deleted = np.array([x for x in sampled if x not in to_delete])
                sequences.append(
                    np.concatenate([deleted, delimiter, to_delete, delimiter, deleted])
                )
            else:
                sampled = self.vocab.sample_vocab(sequence_length // 2 - 1, vocab_rng)
                to_delete = sampled[
                    delete_index : min(len(sampled), delete_index + delete_length)
                ]
                sample_deleted = sampled.tolist()
                for i in range(len(sample_deleted) - delete_length + 1):
                    if all(
                        [
                            a == b
                            for a, b in zip(
                                sample_deleted[i : i + delete_length], to_delete
                            )
                        ]
                    ):
                        for j in range(i, i + delete_length):
                            sample_deleted[j] = -1
                sample_deleted = np.array([x for x in sample_deleted if x != -1])

                sequences.append(
                    np.concatenate(
                        [sampled, delimiter, to_delete, delimiter, sample_deleted]
                    )
                )

        return sequences


if __name__ == "__main__":
    delete = Delete(
        sequence_length=20,
        should_delete_ratio=1,
        seq_to_delete_length=Range(min=2, max=4),
        vocab=Vocabulary(num_vocab=20000),
    )

    seqs = delete.make_sequences(20)
    print(seqs)
