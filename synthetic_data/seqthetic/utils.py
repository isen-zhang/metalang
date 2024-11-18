import re
from typing import Annotated

import numpy as np
from nanoid import generate
from pydantic import BaseModel, BeforeValidator, Field

from seqthetic.range import FlexibleRange


_size_pattern = re.compile(r"[0-9]+(?:\.[0-9]+)?[BMK]")
_id_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
generate_id = lambda: generate(alphabet=_id_alphabet, size=10)
ID = Field(default_factory=generate_id)
DelimiterTokenField = Field(default_factory=lambda: [-1])


def validate_size_value(v):
    if isinstance(v, str):
        v = v.upper()
        scale = 0
        if _size_pattern.match(v):
            if v.endswith("B"):
                scale = 1000000000
            elif v.endswith("M"):
                scale = 1000000
            elif v.endswith("K"):
                scale = 1000
            else:
                raise ValueError("invalid size format")
            return float(v[:-1]) * scale

    elif isinstance(v, int):
        return v
    elif isinstance(v, float):
        return int(v)
    else:
        raise TypeError("string or integer required")


SizeValue = Annotated[str | int, BeforeValidator(validate_size_value)]


class MyModel(BaseModel):
    size: SizeValue


def sample_integer_range(
    length: FlexibleRange, rng: np.random.Generator, num_sequence: int
):
    sequence_lengths = (
        rng.integers(length.min, length.max, num_sequence)
        if not length.constant
        else [int(length.min)] * num_sequence
    )
    return sequence_lengths


def make_digitizer(ratio: float):
    # 从最大到最小值均分
    def digitize(arr):
        bin_count = round(len(arr) // ratio)
        bins = np.linspace(arr.max(), arr.min(), bin_count)[1:-1]
        return np.digitize(arr, bins)

    return digitize


def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def get_seqlen_and_numseq(
    sequence_length: FlexibleRange, num_token: int, seqlen_rng: np.random.Generator
) -> tuple[list[int], int]:
    if sequence_length.constant:
        num_sequence = int(num_token // sequence_length.min)
        sequence_lengths = [int(sequence_length.min)] * num_sequence
    else:
        avg_length = int((sequence_length.max + sequence_length.min) / 2)
        num_sequence = int(num_token // avg_length)
        sequence_lengths = sample_integer_range(
            sequence_length, seqlen_rng, avg_length
        ).tolist()

        num_tokens_sampled = sum(sequence_lengths)
        while num_tokens_sampled < num_token:
            sampled = sample_integer_range(sequence_length, seqlen_rng, 1)
            sequence_lengths.extend(sampled)
            num_tokens_sampled += sampled
        num_sequence = len(sequence_lengths)

    return sequence_lengths, num_sequence
