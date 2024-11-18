from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import pandas as pd
from seqthetic.domains.delete import Delete
from seqthetic.vocabulary import Vocabulary
from seqthetic.domains import DomainSpec
from seqthetic.seed import make_seed
from seqthetic.utils import ID, SizeValue
import random

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from nanoid import generate
from dataclasses import field as DataclassField
import datetime


class BaseWith(BaseModel):

    domains: list[DomainSpec]


class With(BaseWith):
    key: str
    value: Any

    def model_post_init(self, __context):
        for domain in self.domains:
            setattr(domain, self.key, self.value)
        return self


class WithDomainDelimiter(BaseWith):
    delimiter_length: int = 3
    delimiter_vocab_size: int = 9

    def model_post_init(self, __context):
        vocab = "".join([str(i) for i in range(self.delimiter_vocab_size)])
        delimiters = [
            generate(alphabet=vocab, size=self.delimiter_length) for _ in self.domains
        ]
        for domain, delimiter in zip(self.domains, delimiters):
            setattr(domain, "delimiter", True)
            setattr(domain, "delimiter_tokens", [int(el) for el in delimiter])
        return self


class SplitSpec(BaseModel):
    """
    Two advised usages:
    1. shuffle the dataset: most random and homogenous. just set shuffle_dataset to True, it takes precedence over other shuffle settings
    2. preserve domain order, shuffle sequence order in domains: default mode, useful for curriculum learning
    set shuffle_dataset to False, shuffle_domain_order to False, shuffle_domain_sequence to True
    3. shuffle domain order and sequence order: relatively homogenous shuffle_domain_order to True

    """

    # whether to shuffle across all dataset
    shuffle_dataset: bool = False
    shuffle_domain_order: bool = False
    # whether to shuffle the order of the sequences in each domains
    shuffle_domain_sequence: bool = True
    # ratio between train, val, test, counted by number of sequences
    split_ratio: list[float] = Field(default_factory=lambda: [0.8, 0.1, 0.1])

    def get_index(self, num_items: int):
        train_index = int(self.split_ratio[0] * num_items)
        val_index = train_index + int(self.split_ratio[1] * num_items)
        return train_index, val_index

    @field_validator("split_ratio")
    def check_split_ratio(cls, split_ratio):
        if abs(sum(split_ratio) - 1) > 1e-6:
            raise ValueError("sum of split_ratio should be 1")
        return split_ratio


@dataclass
class DatasetResult:
    domains: list[DomainSpec] = DataclassField(default_factory=list)
    domain_sequences: list[list[np.ndarray]] = DataclassField(default_factory=list)
    # results for each domain
    domain_columns: list[dict[str, list[np.ndarray]]] = DataclassField(
        default_factory=list
    )
    configs: dict[str, any] = DataclassField(default_factory=dict)


class BasePostProcessor(BaseModel):
    adds_columns: dict[str, str] = Field(default_factory=dict)
    adds_configs: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, res: DatasetResult) -> DatasetResult:
        return res


class AddLossMask(BasePostProcessor):
    adds_columns: str = {"loss_mask": "mask for the loss"}
    mask_delimiter: bool = False

    def __call__(self, res: DatasetResult) -> DatasetResult:
        for domain, domain_sequence in zip(res.domains, res.domain_sequences):
            loss_masks = []
            for sequence in domain_sequence:
                if hasattr(domain, "delimiter") and hasattr(domain, "delimiter_tokens"):

                    mask = np.zeros_like(sequence, dtype=int)
                    seq_len = len(sequence)
                    delimiter_len = len(domain.delimiter_tokens)

                    # 寻找完整的delimiter_tokens匹配
                    delimiter_indices = []
                    for i in range(seq_len - delimiter_len + 1):
                        if np.array_equal(
                            sequence[i : i + delimiter_len], domain.delimiter_tokens
                        ):
                            delimiter_indices.append(i)

                    # 根据mask_delimiter决定从哪个位置开始设置1

                    start_idx = delimiter_indices[-1]
                    start_idx = start_idx + delimiter_len if self.mask_delimiter else start_idx
                    mask[start_idx:] = 1

                else:
                    mask = np.ones_like(sequence, dtype=int)
                loss_masks.append(mask)
            res.domain_columns.append({"loss_mask": loss_masks})
        return res


class OffsetVocabulary(BasePostProcessor):
    offset: int = 1

    def __call__(self, res: DatasetResult) -> DatasetResult:
        for domain, domain_sequence in zip(res.domains, res.domain_sequences):
            for sequence in domain_sequence:
                sequence += self.offset

        return res


class DatasetSpec(BaseModel):
    id: str = ID
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    name: str = ""
    num_token: SizeValue
    mixture: list[float] | None = None
    split: SplitSpec = Field(default_factory=SplitSpec)
    domains: list[DomainSpec | With | WithDomainDelimiter] = Field(default_factory=list)
    seed: int = Field(default_factory=make_seed)
    postprocessors: list[BasePostProcessor] = Field(default_factory=list)
    file_parts: int = 1
    configs: dict = Field(default_factory=dict)

    @property
    def domain_list(self) -> list[DomainSpec]:
        domains = []
        for domain in self.domains:
            if isinstance(domain, BaseWith):
                domains.extend(domain.domains)
            else:
                domains.append(domain)
        return domains

    @model_validator(mode="after")
    def check_mixture_length(self):
        if not self.mixture:
            self.mixture = [1 / len(self.domain_list) for _ in self.domain_list]
            return

        if abs(sum(self.mixture) - 1) > 1e-6:
            raise ValueError("sum of mixture should be 1")

        if len(self.mixture) != len(self.domain_list):
            raise ValueError("mixture length should be equal to number of domains")

        return self

    @staticmethod
    def load(path) -> tuple[pd.DataFrame, "DatasetSpec"]:
        with open(path, "r") as f:
            data = json.load(f)
            spec = DatasetSpec.model_validate(data)

        if spec.file_parts == 1:
            df = pd.read_csv(f"{spec.name}.csv")
            return df, spec
        else:
            dfs = []
            for i in range(spec.file_parts):
                print(f"Reading {spec.name}.part{i+1}.csv")
                df = pd.read_csv(f"{spec.name}.part{i+1}.csv")
                dfs.append(df)

            return pd.concat(dfs), spec

    def make_dataset(self, path=None, single_thread=False):
        ss = np.random.SeedSequence(self.seed)
        seeds = [s.entropy for s in ss.spawn(len(self.domain_list))]
        domain_sequences = []
        for seed, domain in zip(seeds, self.domain_list):
            if not single_thread:
                sequences = domain.make_sequences_parallel(self.num_token, seed)
            else:
                sequences = domain.make_sequences(num_token=self.num_token, seed=seed)
            domain_sequences.append(sequences)
        dataset_result = DatasetResult(
            domains=self.domain_list, domain_sequences=domain_sequences
        )

        for processor in self.postprocessors:
            dataset_result = processor(dataset_result)

        if path:
            self.save_dataset_result(dataset_result, path, split_parts=1)

        return dataset_result

    def save_dataset_result(
        self,
        result: DatasetResult,
        path,
        split_parts: int = 1,
    ):
        """
        Save DatasetResult to CSV and config files

        Args:
            result: DatasetResult object
            name: Base name for output files
            split_parts: Number of parts to split the data into (default: 1)
        """
        name = Path(path) / (self.name or self.id)
        # First collect all data into lists
        all_data = defaultdict(list)

        for sequences, columns in zip(result.domain_sequences, result.domain_columns):
            all_data["sequences"].extend(sequences)

            for col_name, col_data in columns.items():
                all_data[col_name].extend(col_data)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Save config
        config_path = f"{name}.config.json"
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

        if split_parts == 1:
            # Save single CSV
            df.to_csv(f"{name}.csv", index=False)
        else:
            # Split into multiple parts
            rows_per_part = len(df) // split_parts
            for i in range(split_parts):
                start_idx = i * rows_per_part
                end_idx = start_idx + rows_per_part if i < split_parts - 1 else len(df)
                part_df = df.iloc[start_idx:end_idx]
                part_df.to_csv(f"{name}.part{i+1}.csv", index=False)
