from typing import Any, final, Union
from harl.envs.harl_env_llm import LLMManager, LLMConfig

from harl.envs.sumo.pettingzoo_sumo_env import (
    TEnv,
    TEnvRaw,
    ObsType,
    StateType,
)


def generate_prompt(curr_step: int) -> str:
    is_lane_close_active = curr_step > 20 and curr_step < 5000

    basic_prompt = """
    please give a judgement on which skillset to trigger or nothing should be triggered.

    ## skillset:
    - lane_close: if the lane is closed, trigger this skillset

    ## desired format:
    {
        "skillset": "lane_close"
    }
    or if nothing should be triggered:
    {
        "skillset": "nothing"
    }

    the output should be pure json with no markdown or other text.
    the output should be able to be processed by json.loads() directly without causing errors.

    ## current_situation:
    """
    if is_lane_close_active:
        basic_prompt += "- lane close is active, please trigger lane close skillset"
    else:
        basic_prompt += "- lane close is not active, please trigger nothing"

    return basic_prompt


@final
class PettingZooSumoLLMManager(LLMManager[Any, TEnv, TEnvRaw, ObsType, StateType]):
    def __init__(
        self,
        parent_env: Any,
        env: TEnv,
        real_env: TEnvRaw,
        llm_config: Union[LLMConfig, None],
    ):
        from ..pettingzoo_sumo_llm_env import PettingZooSumoLLMEnv

        self.parent_env: PettingZooSumoLLMEnv = parent_env

        super().__init__(parent_env, env, real_env, llm_config)

    def _from_obs_to_prompt(self, obses: list[ObsType], global_state: StateType) -> str:
        prompt = generate_prompt(self.parent_env.cur_step)
        return prompt

    def _llm_decision_in_env(self, decisions: dict[str, Any]) -> None:
        if decisions["skillset"] == "lane_close":
            self.parent_env.should_use_predefined_signal_phases = True

        # do something here, before step
