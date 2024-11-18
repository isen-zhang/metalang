from .base import BaseDomainSpec
from .deduplicate import Deduplicate
from .delete import Delete
from .duplicate import Duplicate
from .identity import Identity
from .ngram import NGram
from .sort_numbers import SortNumbers
from .sort_by_sequence import SortBySequence
from .fbm import DiscretizedFBM

DomainSpec = (
    Identity | SortNumbers | Deduplicate | Delete | NGram | SortBySequence | Duplicate | DiscretizedFBM
)
