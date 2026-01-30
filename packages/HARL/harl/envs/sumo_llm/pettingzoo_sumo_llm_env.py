import time
from dataclasses import field
from typing import Any, final
from harl.envs.harl_env_llm import HarlEnvWithLLM
from harl.envs.sumo_llm.llm.sumo_llm_manager import PettingZooSumoLLMManager


# from pettingzoo.sisl import multiwalker_v9
from ..sumo.pettingzoo_sumo_env import (
    PettingZooSumoEnv,
    SumoEnvironmentPZWithGlobalState,
)


@final
class PettingZooSumoLLMEnv(PettingZooSumoEnv, HarlEnvWithLLM):
    should_use_predefined_signal_phases: bool = False
    traffic_info: dict[str, Any] = field(
        default_factory=lambda: {
            "A2": [0, 0, 0, 0],
            "B2": [0, 0, 0, 0],
            "now_A2": 0,
            "now_B2": 0,
        }
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_manager = self._init_llm_manager()
        self.llm_frequency = 200
        self.should_use_predefined_signal_phases = False

    def _init_llm_manager(self) -> PettingZooSumoLLMManager:
        return PettingZooSumoLLMManager(self, self.env, self.env.aec_env.env.env, None)

    def reset(self, *args, **kwargs):
        self.should_use_predefined_signal_phases = False
        self.traffic_info = {
            "A2": [0, 0, 0, 0],
            "B2": [0, 0, 0, 0],
            "now_A2": 0,
            "now_B2": 0,
        }
        return super().reset(*args, **kwargs)

    def step(self, actions):
        actions_wrapped = self.wrap(actions.flatten().tolist())
        _override_signal = {
            # "A2": [-1, -1, 25, 15],
            "B2": [-1, -1, 35, -1],
        }

        if self.cur_step % self.llm_frequency == 0:
            print(f"llm at: ts={self.cur_step}")
            sumo_pz_env: SumoEnvironmentPZWithGlobalState = self._get_sumo_pz_env()
            obs = [sumo_pz_env.observe(agent) for agent in self.agents]
            start_time = time.time()
            self.llm_manager.execute_llm(obs, self.global_state)
            end_time = time.time()
            print(f"llm time: {end_time - start_time}")

        import random

        random_initiated = random.random() < 0.5

        if random_initiated:
            self.should_use_predefined_signal_phases = True
        else:
            self.should_use_predefined_signal_phases = False

        if not self.should_use_predefined_signal_phases:
            self.traffic_info = {
                "A2": [0, 0, 0, 0],
                "B2": [0, 0, 0, 0],
                "now_A2": 0,
                "now_B2": 0,
            }
        if self.should_use_predefined_signal_phases:
            for agent in self.agents:
                if agent != "B2":
                    continue
                now_green_phase = self.traffic_info[f"now_{agent}"]
                proposed_next_action = actions_wrapped[agent]
                self.traffic_info[agent][now_green_phase] += self.args.delta_time
                if (
                    self.traffic_info[agent][now_green_phase]
                    <= _override_signal[agent][now_green_phase]
                ):
                    actions_wrapped[agent] = now_green_phase
                    print(
                        f"ts={self.cur_step}, agent={agent}, current_green_phase={now_green_phase}, policy wants {proposed_next_action}, this_has_been: {self.traffic_info[agent][now_green_phase]}, [not allowed] to change since min is {_override_signal[agent][now_green_phase]}"
                    )
                else:
                    # actions_wrapped[agent] = current_green_phase + 1
                    print(
                        f"ts={self.cur_step}, agent={agent}, current_green_phase={now_green_phase}, policy wants {proposed_next_action}, this_has_been: {self.traffic_info[agent][now_green_phase]}, [allowed] to change! change to {proposed_next_action}"
                    )
                    self.traffic_info[agent][now_green_phase] = 0
                    self.traffic_info[f"now_{agent}"] = proposed_next_action
        import numpy as np

        obs, state, reward, terminated, info, available_actions = super().step(
            # actions,
            np.array(self.unwrap(actions_wrapped))
        )

        return obs, state, reward, terminated, info, available_actions
