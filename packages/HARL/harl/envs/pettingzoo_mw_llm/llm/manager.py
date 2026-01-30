from typing import Any, final, Union
from harl.envs.harl_env_llm import LLMManager, LLMConfig

from harl.envs.pettingzoo_mw.pettingzoo_mw_env import (
    TEnv,
    TEnvRaw,
    ObsType,
    StateType,
)
from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import (
    MultiWalkerEnv as _env_move,
)
from prompt_template import PromptTemplate
from .agent import generate_prompt

import rich


@final
class PettingZooMWLLMManager(LLMManager[Any, TEnv, TEnvRaw, ObsType, StateType]):
    def __init__(
        self,
        parent_env: Any,
        env: TEnv,
        real_env: TEnvRaw,
        llm_config: Union[LLMConfig, None],
    ):
        from ..pettingzoo_mw_llm_env import PettingZooMWLLMEnv

        self.parent_env: PettingZooMWLLMEnv
        super().__init__(parent_env, env, real_env, llm_config)

    def _from_obs_to_prompt(
        self, obses: list[ObsType], global_state: StateType
    ) -> Union[tuple[dict[str, str], PromptTemplate], str]:
        llm_obses = obses
        assert isinstance(self.real_env, _env_move)
        lidar_obs = self.real_env.get_thru_lidar_obs()
        llm_lidar_obses = [
            lidar_obs[agent_id] for agent_id in range(self.env.num_agents)
        ]
        target_vs = [
            self.real_env.get_target_v_agent(agent_id)
            for agent_id in range(self.env.num_agents)
        ]
        prompt = generate_prompt(llm_obses, llm_lidar_obses, target_vs)
        return prompt

    def _llm_decision_in_env(self, decisions: dict[str, Any]) -> None:
        target_vs = decisions["target_vs"]
        assert isinstance(self.real_env, _env_move)
        print("[decisions:] ")
        rich.print(decisions)
        for agent_id in range(self.env.num_agents):
            self.real_env.set_t_v_agent(agent_id, target_vs[agent_id])
