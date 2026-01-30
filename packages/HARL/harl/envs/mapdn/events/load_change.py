from dataclasses import dataclass
from typing import Any
import random

import dacite

from .mapdn_event_manager import MapdnEventManager


@dataclass
class LoadChangeEventArgs:
    multiplier: float


class LoadChangeEventManager(MapdnEventManager):
    event_args: LoadChangeEventArgs | None

    def _event_start(self, args: dict[str, Any]) -> None:
        self.event_args = dacite.from_dict(LoadChangeEventArgs, args)

        self._prepare_env()
        self.real_env.powergrid.load["q_mvar"] = (
            self.real_env.powergrid.load["q_mvar"] * self.event_args.multiplier
        )

    def _event_stop(self) -> None:
        # self._prepare_env()

        # assert self.event_args is not None
        # self.real_env.powergrid.load["q_mvar"] = (
        #     self.real_env.powergrid.load["q_mvar"] / self.event_args.multiplier
        # )
        pass

    def _event_random_value(self) -> Any:
        assert self.event_status.args is not None
        assert self.event_status.args.type == "random"
        return {
            "multiplier": random.uniform(
                self.event_status.args.random_lower,
                self.event_status.args.random_upper,
            )
        }
