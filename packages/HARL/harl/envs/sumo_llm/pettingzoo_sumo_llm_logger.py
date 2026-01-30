from harl.envs.sumo.pettingzoo_sumo_logger import PettingZooSumoLogger
from typing_extensions import override
from typing import Literal


class PettingZooSumoLLMLogger(PettingZooSumoLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooSumoLLMLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )

    def init(self, episodes):
        """Initialize the logger."""
        return super().init(episodes)

    @override
    def get_task_name(self) -> Literal["sumo_llm"]:  # type: ignore
        return "sumo_llm"

    def eval_init(self):
        return super().eval_init()

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        return super().eval_per_step(eval_data)

    def eval_log(self, eval_episode):
        return super().eval_log(eval_episode)
