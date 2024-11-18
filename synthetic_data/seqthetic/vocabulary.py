import numpy as np
from pydantic import BaseModel, Field


from typing import Literal

from pydantic import BaseModel


from typing import Literal


class VocabNoise(BaseModel):
    """noise for vocabulary distribution"""

    type: Literal["multiplicative", "additive"]
    level: float


class Zipf(BaseModel):
    alpha: float = 1
    beta: float = 2.7


class Vocabulary(BaseModel):
    num_vocab: int
    prob: Zipf | Literal["uniform", "corpus"] = "uniform"
    noise: VocabNoise | None = None
    vocab_prob: np.ndarray | None = Field(None, exclude=True)

    class Config:
        # Example of configuring serialization/exclusion
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        self.get_vocab_prob()
        return self

    def get_vocab_prob(self):
        vocab = np.arange(self.num_vocab)
        if isinstance(self.prob, Zipf):
            freq = 1 / (vocab + 1 + self.prob.beta)
            prob = freq / np.sum(freq)
        elif self.prob == "uniform":
            prob = np.ones(self.num_vocab) / self.num_vocab
        self.vocab_prob = np.zeros(
            self.num_vocab, dtype=[("word", np.int32), ("prob", np.float32)]
        )
        self.vocab_prob["word"] = vocab
        self.vocab_prob["prob"] = prob

    def sample_vocab(
        self,
        num_vocab: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        sampled = rng.choice(
            self.vocab_prob,
            size=int(num_vocab),
            replace=False,
            p=self.vocab_prob["prob"],
        )

        return sampled["word"]
