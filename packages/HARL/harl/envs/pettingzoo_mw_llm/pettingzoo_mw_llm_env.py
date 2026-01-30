import logging
from typing import Any, final, Union

# from pettingzoo.sisl import multiwalker_v9
from ..pettingzoo_mw.pettingzoo_mw_env import (
    PettingZooMWEnv,
    ActionType,
    ObsType,
    StateType,
    AllAgentActionAvailableType,
)
from ..harl_env_llm import HarlEnvWithLLM
from .llm.manager import PettingZooMWLLMManager

import time

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)


@final
class PettingZooMWLLMEnv(PettingZooMWEnv, HarlEnvWithLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_manager = self._init_llm_manager()
        self.llm_frequency = 50
        self.llm_manager._llm_decision_in_env(
            {"explanation": "test", "target_vs": [0.8, 0.8, 0.8]}
        )

    def _init_llm_manager(self) -> PettingZooMWLLMManager:
        return PettingZooMWLLMManager(self, self.env, self.raw_env.env, None)

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
        obs, state, reward, terminated, info, available_actions = super().step(actions)
        if self.cur_step % self.llm_frequency == 0:
            start_time = time.time()

            original_prompt: str = self.llm_manager._from_obs_to_prompt(obs, state[0])  # type: ignore
            self.llm_manager._llm_decision_in_env(
                {"explanation": "test", "target_vs": [0.8, 0.8, 0.8]}
            )

            # import random
            # should_use_llm = random.random() < 0.2
            # if should_use_llm:
            if "坡道检测：正在上坡" in original_prompt:
                print(f"---------------Step {self.cur_step}-----------------")
                # print(f"original_prompt: {original_prompt}")
                self.llm_manager.execute_llm(obs, state[0])
                end_time = time.time()
                print(f"llm time: {end_time - start_time}")
        return obs, state, reward, terminated, info, available_actions
