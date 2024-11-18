from abc import ABC
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    PlainSerializer,
    computed_field,
    model_validator,
)


class Range(BaseModel, ABC):
    min: float
    max: float

    @computed_field
    @property
    def constant(self) -> bool:
        return abs(self.min - self.max) < 1e-6

    @model_validator(mode="after")
    def validate_range(self):
        if self.min <= 0 or self.max <= 0:
            raise ValueError("value should be positive")
        elif self.min > self.max:
            raise ValueError("min should be less than max")

        return self

    def __str__(self):
        if self.constant:
            return f"{self.min}"
        else:
            return f"Range({self.min}, {self.max})"


FlexibleRangeInput = float | int | Range


def convert_single_number(v: FlexibleRangeInput):
    if isinstance(v, int) or isinstance(v, float):
        if v <= 0:
            raise ValueError("value should be positive")
        return Range(min=v, max=v)
    return v


def serialize_flexible_range(v: Range):
    if isinstance(v, int) or isinstance(v, float):
        return v
    if v.constant:
        if v.min % 1 != 0:
            return v.min
        else:
            return int(v.min)
    else:
        return {"min": v.min, "max": v.max}


FlexibleRange = Annotated[
    FlexibleRangeInput,
    AfterValidator(convert_single_number),
    PlainSerializer(serialize_flexible_range),
]
