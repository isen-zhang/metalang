from pydantic import Field
from seqthetic.domains.base import BaseDomainSpec
from seqthetic.range import FlexibleRange
from typing import Literal, TypedDict
import numpy as np
from seqthetic.utils import get_seqlen_and_numseq, DelimiterTokenField
from seqthetic.vocabulary import Vocabulary


class SuffixPosition(TypedDict):
    prefix: bool = True
    affix: bool = True
    suffix: bool = True


class FindArrayElement(BaseDomainSpec):
    """
    Given a list of elements of various lengths and a substring, find the element that contains the substring.

    Input: a b c <sep> d e f <sep> g h i
    Target: e f
    Output: d e f

    """
    type: Literal["find_array_element"] = "find_array_element"
    sequence_length: FlexibleRange
    element_count: FlexibleRange
    element_length: FlexibleRange
    suffix_ratio: float = Field(gt=0, ge=1)
    suffix_position: SuffixPosition
    vocab: Vocabulary
    delimiter: bool = True
    delimiter_tokens: list[int] = DelimiterTokenField

    def make_sequences(self, num_token, seed):
        sequence_lengths, num_sequence = get_seqlen_and_numseq(
            self.sequence_length, num_token, sequence_length
        )
        sequences = []
        for _ in range(num_sequence):
            elements = [
                self.vocab.sample_vocab(self.element_length)
                for _ in range(self.element_count)
            ]
            found_element = np.sample(elements)
            
