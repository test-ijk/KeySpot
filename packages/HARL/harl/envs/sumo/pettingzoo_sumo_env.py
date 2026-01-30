import copy
import logging
import time

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from sumo_rl.environment.env import SumoEnvironment
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import sumo_rl
import numpy as np
import random

import gymnasium as gym

# from pettingzoo.sisl import multiwalker_v9
from ..harl_env_with_events import (
    HarlEnvWithEvents,
    Event,
    EnvProtocol,
)

# from pettingzoo.sisl import multiwalker_v9

logging.basicConfig(level=logging.WARNING)


class SumoEnvironmentPZWithGlobalState(sumo_rl.SumoEnvironmentPZ, EnvProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_space = self.get_state_space()

    def get_state_space(self):
        _a = self.env.ts_ids[0]
        ts = self.env.traffic_signals[_a]
        low = ts.num_green_phases + 1 + 2 * len(ts.lanes)
        return spaces.Box(
            low=np.zeros(low * self.num_agents, dtype=np.float32),
            high=np.ones(low * self.num_agents, dtype=np.float32),
        )

    def state(self):
        obs = []
        for agent in self.agents:
            obs.append(self.observe(agent))
        global_state = np.array(obs).flatten().astype(np.float32)
        return global_state

    def step(self, action):
        super().step(action)


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZWithGlobalState(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


@dataclass
class SumoEnvConfig:
    net_file: str
    route_file: str
    out_csv_name: Optional[str] = None
    use_gui: bool = False
    virtual_display: list[int] = field(default_factory=lambda: [3200, 1800])
    begin_time: int = 0
    num_seconds: int = 20000
    max_depart_delay: int = -1
    waiting_time_memory: int = 1000
    time_to_teleport: int = -1
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50
    enforce_max_green: bool = False
    single_agent: bool = False
    reward_fn: str = "diff-waiting-time"
    reward_weights: Optional[list[float]] = None

    observation_class: Union[str, type] = "DefaultObservationFunction"

    add_system_info: bool = True
    add_per_agent_info: bool = True
    sumo_seed: Union[str, int] = "random"
    fixed_ts: bool = False
    sumo_warnings: bool = True
    additional_sumo_cmd: Optional[str] = None
    render_mode: Optional[str] = None
    
    action_perturb_prob: float = 1
    perturb_targets: list[str] = field(default_factory=list)

    events: Optional[list[Event]] = None


ActionType = np.ndarray[Any, np.dtype[np.int32]]
ObsType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]
StateType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]


TAgentId = str
TEnv = aec_to_parallel_wrapper[str, ObsType, ActionType]
TArgs = dict[str, Any]
TDeepDict = dict[TAgentId, dict[str, Any]]
ObsWrappedType = dict[TAgentId, ObsType]
TEnvRaw = SumoEnvironment


class PettingZooSumoEnv(
    HarlEnvWithEvents[
        str,
        aec_to_parallel_wrapper,
        SumoEnvConfig,
        ObsType,
        ActionType,
        StateType,
        SumoEnvironment,
    ]
):
    events: list[Event]
    n_agents: int
    share_observation_space: list[gym.spaces.Box]
    observation_space: list[gym.spaces.Box]
    action_space: list[Union[gym.spaces.Box, gym.spaces.Discrete]]
    agents: list[TAgentId]
    _seed: int

    def __init__(self, args: SumoEnvConfig):
        self.args: SumoEnvConfig = copy.deepcopy(args)

        args.observation_class = sumo_rl.DefaultObservationFunction

        self.discrete: bool = True

        self.max_cycles: int = (args.num_seconds) // args.delta_time - 1

        self.cur_step: int = 0

        dict_args = asdict(args)
        dict_args["virtual_display"] = tuple(dict_args["virtual_display"])
        del dict_args["observation_class"]
        del dict_args["events"]

        self.action_perturb_prob = getattr(args, "action_perturb_prob", 1)
        targets = getattr(args, "perturb_targets", [''])
        if targets is None:
            self.perturb_targets = None
        else:
            self.perturb_targets = set(targets)
        self._rng = np.random.RandomState(getattr(args, "seed", 0))
        dict_args.pop("action_perturb_prob", None)
        dict_args.pop("perturb_targets", None)

        self.env = parallel_env(**dict_args, observation_class=args.observation_class)
        self.env.reset()

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents

        self.observation_space = self.unwrap(self.env.observation_spaces)  # type: ignore
        self.action_space = self.unwrap(self.env.action_spaces)  # type: ignore
        self.share_observation_space = self.repeat(self.env.state_space)
        self._seed = 0
        self.cur_step = 0


        # events
        if args.events is not None:
            self._init_event_mapping()
            self._init_event(args.events)

        super().__init__(args)

    def _get_real_env(self) -> TEnvRaw:
        a_env: SumoEnvironmentPZWithGlobalState = self.env.aec_env.env.env
        return a_env.env

    def _get_sumo_pz_env(self) -> SumoEnvironmentPZWithGlobalState:
        return self.env.aec_env.env.env

    def _init_event_mapping(self) -> None:
        from .events.lane_closed import LaneCloseEventManager

        self.event_mapping = {
            "lane_closed": LaneCloseEventManager,
        }

    @property
    def global_state(self) -> StateType:
        return self.env.state()

    def action_disturb(self, actions):
        if self.action_perturb_prob <= 0.0 or not self.perturb_targets:
            return actions

        for tl_id, old_a in actions.items():
            if tl_id in self.perturb_targets:
                if random.random() < self.action_perturb_prob:
                    # print(f"action_perturb_prob:{self.action_perturb_prob}")
                    candidates = [0, 1, 2, 3]
                    new_a = random.choice(candidates)
                    actions[tl_id] = new_a
        return actions

    def obs_disturb(self, obs):
        disturbed_obs = {}

        for agent, o in obs.items():
            arr = np.asarray(o, dtype=np.float32).copy()

            if agent not in self.perturb_targets:
                disturbed_obs[agent] = arr
                continue
            
            if arr.shape[0] < 6:
                disturbed_obs[agent] = arr
                continue

            # noise_dim = arr.shape[0] - 5
            # noise = np.random.normal(
            #     loc=0.0,
            #     scale=0.1,
            #     size=noise_dim,
            # ).astype(np.float32)

            # arr[5:] = arr[5:] + noise
            arr[5:] = arr[5:] * 8.0
            arr[5:] = np.clip(arr[5:], 0.0, 1.0)

            disturbed_obs[agent] = arr

        return disturbed_obs

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        actions_wrapped = self.wrap(actions.flatten().tolist())
        if self.cur_step > 100 and self.cur_step <200:
            actions_wrapped = self.action_disturb(actions_wrapped)

        obs, rew, term, trunc, info = self.env.step(actions_wrapped)  # type: ignore


        self.cur_step += 1
        info["step"] = self.cur_step
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True

        for agent in self.agents:
            info[agent]["curr_step"] = self.cur_step

        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        total_reward: float = sum([rew[agent] for agent in self.agents])
        rewards: list[list[float]] = [[total_reward]] * self.n_agents


        # if self.cur_step > 100 and self.cur_step <200:
        #     obs = self.obs_disturb(obs)
        self.trigger_event()
        return (
            self.unwrap(obs),
            self.repeat(self.global_state),
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        obs, infos = self.env.reset(seed=self._seed)  # type: ignore
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.global_state)
        return obs, s_obs, self.get_avail_actions()

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed
