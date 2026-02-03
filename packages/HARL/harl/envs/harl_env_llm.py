from typing import Union, Any, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .harl_env_with_events import (
    HarlEnvWithEvents,
    TAgentId,
    TEnv,
    TArgs,
    ObsType,
    ActionType,
    StateType,
    AllAgentActionAvailableType,
    TEnvRaw,
)
from openai import OpenAI
from dotenv import load_dotenv
import os
from prompt_template import PromptTemplate


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str


TParentEnv = TypeVar("TParentEnv", covariant=True)


class LLMManager(ABC, Generic[TParentEnv, TEnv, TEnvRaw, ObsType, StateType]):
    parent_env: TParentEnv
    env: TEnv
    real_env: TEnvRaw
    llm_client: OpenAI
    llm_config: LLMConfig
    prompt_template_library: dict[str, PromptTemplate]

    def __init__(
        self,
        parent_env: TParentEnv,
        env: TEnv,
        real_env: TEnvRaw,
        llm_config: Union[LLMConfig, None] = None,
    ):
        self.parent_env = parent_env
        self.env = env
        self.real_env = real_env
        if llm_config is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY is not set")
            if model is None:
                raise ValueError("OPENAI_MODEL is not set")
            self.llm_config = LLMConfig(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            self.llm_config = llm_config
        self.llm_client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
        )

    def get_llm_decisions(
        self, obses: list[ObsType], global_state: StateType
    ) -> dict[str, Any]:

        import json

        result = self.get_llm_result(obses, global_state)
        try:
            json_loaded = json.loads(result)
            assert isinstance(json_loaded, dict)
            return json_loaded
        except Exception:
            print(f"Error: {result}")
            return {"target_vs": [0.4, 0.4, 0.4]}

    def translate_obses(
        self, obses: list[ObsType], global_state: StateType
    ) -> tuple[str, dict[str, str], Union[PromptTemplate, None]]:
        translate_result = self._from_obs_to_prompt(obses, global_state)
        if isinstance(translate_result, tuple):
            semantic_infos, prompt_template = translate_result
            return (
                self.use_semantic_info_to_build_prompt(prompt_template, semantic_infos),
                semantic_infos,
                prompt_template,
            )
        else:
            return translate_result, {}, None

    def use_semantic_info_to_build_prompt(
        self, prompt_template: PromptTemplate, semantic_infos: dict[str, str]
    ) -> str:
        return prompt_template.to_string(**semantic_infos)

    def get_llm_result(self, obses: list[ObsType], global_state: StateType) -> str:
        prompt, semantic_infos, prompt_template = self.translate_obses(
            obses, global_state
        )
        # print(f"prompt: {prompt}")
        return self.get_llm_response(prompt)

    def get_llm_response(self, prompt: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = response.choices[0].message.content
            assert result is not None, "llm response is None"
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_llm(self, obses: list[ObsType], global_state: StateType) -> None:
        decisions = self.get_llm_decisions(obses, global_state)
        self._llm_decision_in_env(decisions)

    @abstractmethod
    def _from_obs_to_prompt(
        self, obses: list[ObsType], global_state: StateType
    ) -> Union[tuple[dict[str, str], PromptTemplate], str]:
        """

        Args:
            obses: list[ObsType]
            global_state: StateType

        Returns:
            Union[tuple[dict[str, str], PromptTemplate], str]:
                If tuple, return semantic_infos and prompt_template
                If str, return prompt
        """
        pass

    @abstractmethod
    def _llm_decision_in_env(self, decisions: dict[str, Any]) -> None:
        pass


class HarlEnvWithLLM(
    HarlEnvWithEvents[TAgentId, TEnv, TArgs, ObsType, ActionType, StateType, TEnvRaw],
    ABC,
):
    llm_manager: LLMManager[HarlEnvWithEvents, TEnv, TEnvRaw, ObsType, StateType]
    llm_frequency: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_manager = self._init_llm_manager()
        self.llm_frequency = 50

    @abstractmethod
    def _init_llm_manager(
        self,
    ) -> LLMManager[HarlEnvWithEvents, TEnv, TEnvRaw, ObsType, StateType]:
        pass

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
            self.llm_manager.execute_llm(obs, state[0])
        return obs, state, reward, terminated, info, available_actions
