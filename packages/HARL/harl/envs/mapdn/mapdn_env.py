import copy
import logging

from dataclasses import asdict, dataclass, field
from typing import Any, Union, Literal, TypeVar
from gymnasium import spaces
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
import numpy as np

import gymnasium as gym

# from pettingzoo.sisl import multiwalker_v9
from ..harl_env_with_events import HarlEnvWithEvents, Event, EnvProtocol

# from pettingzoo.sisl import multiwalker_v9

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


@dataclass
class MapdnEnvConfig:


    voltage_barrier_type: Literal["l1", "l2", "bowl", "courant_beltrami", "bump"]
    voltage_weight: float = 1.0
    q_weight: float = 0.1
    line_weight: Union[float, None] = None
    dq_dv_weight: Union[float, None] = None
    history: int = 1
    pv_scale: float = 1.0
    demand_scale: float = 1.0
    state_space: list[str] = field(
        default_factory=lambda: ["pv", "demand", "reactive", "vm_pu", "va_degree"]
    )
    v_upper: float = 1.05
    v_lower: float = 0.95
    data_path: str = (
        "./packages/MAPDN/mapdn/environments/var_voltage_control/data/case33_3min_final"
    )
    episode_limit: int = 240
    action_scale: Union[float, None] = None
    action_bias: Union[float, None] = None
    mode: Union[Literal["distributed", "decentralised"], None] = "distributed"
    reset_action: bool = True
    seed: int = 42
    max_cycles: int = 240

    events: Union[list[Event], None] = None

    scenario: Literal[
        "case33_3min_final",
        "case141_3min_final",
        "case322_3min_final",
    ] = "case33_3min_final"

    test_day: int = 180
    test_hour: int = 23
    test_quarter: int = 2

    def __post_init__(self):
        if self.data_path[-1] != "/":
            self.data_path += "/"


ActionType = np.ndarray[Any, np.dtype[np.int32]]
ObsType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]
StateType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]

TAgentId = str
TArgs = dict[str, Any]
TDeepDict = dict[TAgentId, dict[str, Any]]
ObsWrappedType = dict[TAgentId, ObsType]

T = TypeVar("T")


class MapdnWrapperEnv(EnvProtocol):
    def __init__(self, args: MapdnEnvConfig, disturbances: list[Any]):
        self.args: MapdnEnvConfig = copy.deepcopy(args)
        dict_args = asdict(args)
        del dict_args["events"]
        self.real_env = VoltageControl(dict_args, disturbances)
        self.n_agents = self.real_env.n_agents
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

    def close(self):
        pass

    def observation_spaces(self) -> dict[TAgentId, gym.spaces.Box]:
        return {agent: self.observation_space() for agent in self.agents}

    def observation_space(self) -> gym.spaces.Box:
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.real_env.get_obs_size(),),
            dtype=np.float32,
        )

    def action_spaces(self) -> dict[TAgentId, gym.spaces.Box]:
        return {agent: self.action_space() for agent in self.agents}

    def action_space(self) -> gym.spaces.Box:
        return spaces.Box(
            low=np.float32(
                -self.real_env.args.action_scale + self.real_env.args.action_bias
            ),
            high=np.float32(
                self.real_env.args.action_scale + self.real_env.args.action_bias
            ),
            shape=(1,),
            dtype=np.float32,
        )

    def global_state_space(self) -> gym.spaces.Box:
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.real_env.get_state_size(),),
            dtype=np.float32,
        )

    def state(self) -> StateType:
        return self.real_env.get_state()

    def _type_safe_get_obs(self) -> list[ObsType]:
        return self.real_env.get_obs()

    def _type_safe_env_step(
        self, actions: ActionType
    ) -> tuple[list[ObsType], float, bool, dict[str, Any]]:
        reward, terminated, info = self.real_env.step(actions)  # type: ignore
        reward: float
        terminated: bool
        info: dict[str, Any]
        obs = self._type_safe_get_obs()
        return obs, reward, terminated, info

    

    def step(self, actions: ActionType):

        obs, reward, terminated, info = self._type_safe_env_step(actions)  # type: ignore

        return (
            self.wrap(obs),
            self.wrap(self.repeat(reward)),
            self.wrap(self.repeat(terminated)),
            self.wrap(self.repeat(terminated)),
            self.wrap(self.repeat(info)),
        )


    def reset(self, seed: Union[int, None] = None):
        obs, global_state = self.real_env.reset()
        return self.wrap(obs), global_state

    def render(self) -> None:
        self.real_env.render()

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


ActionAvailableType = list[int]
AllAgentActionAvailableType = list[ActionAvailableType]


class MapdnHARLEnv(
    HarlEnvWithEvents[
        TAgentId,
        MapdnWrapperEnv,
        MapdnEnvConfig,
        ObsType,
        ActionType,
        StateType,
        VoltageControl,
    ]
):
    events: list[Event]
    n_agents: int
    share_observation_space: list[gym.spaces.Box]
    observation_space: list[gym.spaces.Box]
    action_space: list[Union[gym.spaces.Box, gym.spaces.Discrete]]
    agents: list[TAgentId]
    _seed: int

    def __init__(self, args: MapdnEnvConfig, is_eval: bool = False):
        self.args: MapdnEnvConfig = copy.deepcopy(args)

        self.discrete: bool = False
        self.max_cycles: int = args.max_cycles
        self.cur_step: int = 0

        self.env = MapdnWrapperEnv(args, disturbances=[])
        self.env.reset()

        self.n_agents = self.env.n_agents
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.observation_space = self.unwrap(self.env.observation_spaces())  # type: ignore
        self.action_space = self.unwrap(self.env.action_spaces())  # type: ignore
        self.share_observation_space = self.repeat(self.env.global_state_space())
        self._seed = 0
        self.cur_step = 0

        # events
        if args.events is not None:
            self._init_event_mapping()
            self._init_event(args.events)

        super().__init__(args)

        # if is_eval:
        #     self.env.real_env.manual_reset(
        #         args.test_day, args.test_hour, args.test_quarter
        #     )
        #     print("if this got printed, then the manual_reset is called")

    def _init_event_mapping(self) -> None:
        from .events.load_change import LoadChangeEventManager

        self.event_mapping = {
            "load_change": LoadChangeEventManager,
        }

    @property
    def global_state(self) -> StateType:
        return self.env.state()

    def step(
        self, actions
    ) -> tuple[
        list[ObsType],
        list[StateType],
        list[list[float]],
        list[bool],
        list[dict[str, Any]],
        Union[AllAgentActionAvailableType, None],
    ]:
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        self.cur_step += 1
        acts = actions.flatten().tolist()

        # should_use_llm = random.random() < 0.2
        # if should_use_llm:
        # acts = [1] * self.n_agents
        # if self.cur_step >= 101 and self.cur_step <= 200:
        #     acts = [1] * self.n_agents
        obs, rew, term, trunc, info = self.env.step(acts)  # type: ignore

        for agent in self.agents:
            info[agent]["curr_step"] = self.cur_step
            if self.cur_step == self.max_cycles:
                trunc = {agent: True for agent in self.agents}
                info[agent]["bad_transition"] = True

        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        global_state = self.wrap(self.repeat(self.global_state))
        total_reward: float = sum([rew[agent] for agent in self.agents])
        rewards: list[list[float]] = [[total_reward]] * self.n_agents
        self.trigger_event()

        return (
            self.unwrap(obs),
            self.unwrap(global_state),
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        self.seed(self._seed)
        obs, global_state = self.env.reset(seed=self._seed)  # type: ignore
        obs = self.unwrap(obs)
        s_obs = self.repeat(global_state)
        return obs, s_obs, self.get_avail_actions()

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed
        from harl.utils.envs_tools import set_seed

        set_seed({"seed_specify": True, "seed": seed})
