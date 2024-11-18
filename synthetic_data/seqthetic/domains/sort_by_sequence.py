from seqthetic.domains.base import BaseDomainSpec
from seqthetic.utils import DelimiterTokenField

class SortBySequence(BaseDomainSpec):
    delimiter: bool = True
    delimiter_tokens: list[int] = DelimiterTokenField
    pass
