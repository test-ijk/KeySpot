from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, Union, Any, cast
from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np


@dataclass
class EventData_GivenValue:
    type: Literal["given"]
    given_value: dict[str, Any]


@dataclass
class EventData_RandomValue:
    type: Literal["random"]
    random_upper: float
    random_lower: float
    random_method: Literal["uniform"] = "uniform"


@dataclass
class GivenTimestepsTriggerArgs:
    trigger_at_timestep: int
    stop_at_timestep: int
    event_value: Union[EventData_GivenValue, EventData_RandomValue]


@dataclass
class RandomTriggerArgs:
    trigger_frequency: float
    event_value: Union[EventData_GivenValue, EventData_RandomValue]
    duration: int


@dataclass
class Event:
    event_id: str
    should_trigger_by_given_timestep: bool
    given_timestep_trigger_args: Union[GivenTimestepsTriggerArgs, None]
    should_trigger_by_random: bool
    random_trigger_args: Union[RandomTriggerArgs, None]
    lasting: bool = False

    def __post_init__(self):
        if (
            self.should_trigger_by_given_timestep
            and self.given_timestep_trigger_args is None
        ):
            raise ValueError(
                "given_timestep_trigger_args must be provided when should_trigger_by_given_timestep is true"
            )
        if self.should_trigger_by_random and self.random_trigger_args is None:
            raise ValueError(
                "random_trigger_args must be provided when should_trigger_by_random is true"
            )


class EnvProtocol(Protocol):
    def reset(self, *args, **kwargs) -> Any: ...
    def step(self, *args, **kwargs) -> Any: ...
    def close(self) -> None: ...
    def state(self) -> Any: ...
    def render(self, *args, **kwargs) -> Any: ...


TEnv = TypeVar("TEnv", bound=EnvProtocol)

TAgentId = TypeVar("TAgentId")
TArgs = TypeVar("TArgs")
T = TypeVar("T")
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType", contravariant=True)
StateType = TypeVar("StateType")

ActionAvailableType = list[int]
AllAgentActionAvailableType = list[ActionAvailableType]


@dataclass
class EventStatus:
    is_active: bool
    started_at: int
    stopped_at: int
    args: Union[EventData_GivenValue, EventData_RandomValue, None]


TEnvRaw = TypeVar("TEnvRaw")


class EventManager(Generic[TEnv, TEnvRaw], ABC):
    event_config: Event
    event_status: EventStatus
    original_backup: Any
    env: TEnv
    real_env: TEnvRaw
    event_args: Any | None

    def _get_real_env(self) -> TEnvRaw:
        return self.real_env

    @abstractmethod
    def _extract_real_env(self) -> TEnvRaw:
        pass

    def __init__(self, event_config: Event, env: TEnv):
        self.event_config = event_config
        self.event_status = EventStatus(
            is_active=False, started_at=0, stopped_at=0, args=None
        )
        self.env = env
        self.real_env = self._extract_real_env()

    @abstractmethod
    def _event_random_value(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _event_start(self, args: Any) -> None:
        """
        _get_event_args_value()
        """
        pass

    @abstractmethod
    def _event_stop(self) -> None:
        pass

    def _prepare_env(self) -> None:
        self.real_env = self._extract_real_env()
        assert self.real_env is not None

    def _get_event_args_value(self) -> Any | None:
        if self.event_status.args is None:
            raise ValueError("args must be provided to be triggered")
        if self.event_status.args.type == "given":
            self.event_args = self.event_status.args.given_value
            return self.event_args
        elif self.event_status.args.type == "random":
            self.event_args = self._event_random_value()
            return self.event_args

    def start(self, cur_step: int) -> None:
        event_status = self.event_status
        event = self.event_config
        if event_status.is_active and not event.lasting:
            print(f"Event {event.event_id} is already active")
            return
        event_status.is_active = True
        event_status.started_at = cur_step
        if event.should_trigger_by_given_timestep:

            assert event.given_timestep_trigger_args is not None, (
                "given_timestep_trigger_args must be provided to be triggered"
            )
            if event.given_timestep_trigger_args.event_value.type == "given":
                event_status.args = event.given_timestep_trigger_args.event_value
            elif event.given_timestep_trigger_args.event_value.type == "random":
                event_status.args = event.given_timestep_trigger_args.event_value
            event_status.stopped_at = event.given_timestep_trigger_args.stop_at_timestep
        elif event.should_trigger_by_random:
            assert event.random_trigger_args is not None, (
                "random_trigger_args must be provided to be triggered"
            )
            if event.random_trigger_args.event_value.type == "given":
                event_status.args = event.random_trigger_args.event_value
            elif event.random_trigger_args.event_value.type == "random":
                event_status.args = event.random_trigger_args.event_value
            event_status.stopped_at = cur_step + event.random_trigger_args.duration
        assert event_status.args is not None, "args must be provided to be triggered"

        self._event_start(self._get_event_args_value())

    def stop(self) -> None:
        self._event_stop()


class HarlEnvWithEvents(
    Generic[TAgentId, TEnv, TArgs, ObsType, ActionType, StateType, TEnvRaw], ABC
):
    events: list[Event]
    event_managers: list[EventManager[TEnv, TEnvRaw]]
    event_mapping: dict[str, type[EventManager[TEnv, TEnvRaw]]]

    max_cycles: int
    discrete: bool
    n_agents: int
    share_observation_space: list[gym.spaces.Box]
    observation_space: list[gym.spaces.Box]
    action_space: list[Union[gym.spaces.Box, gym.spaces.Discrete]]
    cur_step: int
    agents: list[TAgentId]
    env: TEnv
    args: TArgs
    _seed: int

    def __init__(self, args):
        some_must_set = [
            "n_agents",
            "agents",
            "share_observation_space",
            "observation_space",
            "action_space",
            "max_cycles",
            "discrete",
            "env",
            "cur_step",
            "args",
        ]
        for attr in some_must_set:
            if not hasattr(self, attr):
                raise NotImplementedError(f"subclass must set {attr} in __init__")
        _ = self.reset()
        self.cur_step = 0

    @abstractmethod
    def step(
        self, actions: ActionType
    ) -> tuple[
        list[ObsType],
        list[StateType],
        list[list[float]],
        list[bool],
        list[dict[str, Any]],
        Union[AllAgentActionAvailableType, None],
    ]:
        pass

    @abstractmethod
    def reset(
        self,
    ) -> tuple[
        list[ObsType], list[StateType], Union[AllAgentActionAvailableType, None]
    ]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def render(self) -> Union[np.ndarray[Any, np.dtype[np.uint8]], None]:
        pass

    def get_avail_actions(self) -> Union[AllAgentActionAvailableType, None]:
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_idx: int) -> ActionAvailableType:
        # Same for everyone: return [1] * self.action_space[agent_id].n
        if isinstance(self.action_space[agent_idx], gym.spaces.Discrete):
            space = cast(gym.spaces.Discrete, self.action_space[agent_idx])
            action_a = space.n
            return [1] * action_a
        else:
            space = cast(gym.spaces.Box, self.action_space[agent_idx])
            assert space.shape is not None
            return [1] * space.shape[0]

    def wrap(self, lam: list[T]) -> dict[TAgentId, T]:

        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = lam[i]
        return d

    def unwrap(self, d: dict[TAgentId, T]) -> list[T]:
        _tmp = []
        for agent in self.agents:
            _tmp.append(d[agent])
        return _tmp

    def repeat(self, a: T) -> list[T]:
        return [a for _ in range(self.n_agents)]

    @abstractmethod
    def seed(self, seed: int) -> None:
        pass

    @abstractmethod
    def _init_event_mapping(self) -> None:
        pass

    def _init_event(self, events: list[Event]) -> None:
        assert self.event_mapping is not None, "event_mapping must be provided"
        self.events = events
        self.event_managers = [
            self.event_mapping[event.event_id](event_config=event, env=self.env)
            for event in self.events
        ]

    def get_events(self) -> list[Event]:
        return self.events

    def get_event_managers(self) -> list[EventManager[TEnv, TEnvRaw]]:
        return self.event_managers

    def trigger_event(self) -> None:
        if self.events is None:
            print("[WARNING] events is None, no events to be triggered")
            return
        assert self.event_mapping is not None, (
            "events provided, however event_mapping is not provided"
        )
        for event_idx, event_manager in enumerate(self.event_managers):
            if event_manager.event_config.should_trigger_by_given_timestep:
                assert (
                    event_manager.event_config.given_timestep_trigger_args is not None
                ), (
                    f"{event_idx} given_timestep_trigger_args must be provided to be triggered"
                )
                if event_manager.event_config.lasting:
                    if (
                        self.cur_step
                        >= event_manager.event_config.given_timestep_trigger_args.trigger_at_timestep
                        and self.cur_step
                        <= event_manager.event_config.given_timestep_trigger_args.stop_at_timestep
                    ):
                        event_manager.start(self.cur_step)
                else:
                    if (
                        self.cur_step
                        == event_manager.event_config.given_timestep_trigger_args.trigger_at_timestep
                    ):
                        event_manager.start(self.cur_step)
            elif event_manager.event_config.should_trigger_by_random:
                import random

                assert event_manager.event_config.random_trigger_args is not None, (
                    f"{event_idx} random_trigger_args must be provided to be triggered"
                )
                random_value_trigger = random.random()
                if (
                    random_value_trigger
                    <= event_manager.event_config.random_trigger_args.trigger_frequency
                ):
                    event_manager.start(self.cur_step)
            if event_manager.event_status.is_active:
                if self.cur_step == event_manager.event_status.stopped_at:
                    event_manager.stop()
