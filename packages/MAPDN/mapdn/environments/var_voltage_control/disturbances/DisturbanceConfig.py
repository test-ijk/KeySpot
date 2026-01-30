from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class DisturbanceType(Enum):
    load_change = "load_change"


@dataclass
class DisturbanceConfig:
    type: DisturbanceType

    # for pre-defined disturbance
    start_at: int
    end_at: int

    disturbance_args: Any

    # for random disturbance
    is_random: bool = False
    random_probability: Optional[int] = 0.02
    random_duration: Optional[int] = 200
