from harl.common.base_logger import BaseLogger
import numpy as np


class PettingZooMWLLMLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooMWLLMLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False
        self.test_data = {
            "terminate_at": [],
            "angle_data": [
                [] for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"])
            ],
            "package_x": [],
            "v_deviation": [],
        }

    def init(self, episodes):
        """Initialize the logger."""
        super().init(episodes)
        self.test_data = {
            "terminate_at": [],
            "angle_data": [
                [] for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"])
            ],
            "package_x": [],
            "v_deviation": [],
        }

    def get_task_name(self):
        return "mw_pettingzoo"

    def eval_init(self):
        super().eval_init()
        self.test_data = {
            "terminate_at": [],
            "angle_data": [
                [] for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"])
            ],
            "package_x": [],
            "v_deviation": [],
        }

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        if not self.test_data:
            super().eval_per_step(eval_data)
        else:
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = eval_data
            for i in range(len(eval_infos)):
                self.test_data["angle_data"][i].append(
                    eval_infos[i][0]["package_angle"]
                )
            for i in range(len(eval_infos)):
                if eval_dones[i][0]:
                    self.test_data["terminate_at"].append(eval_infos[i][0]["curr_step"])
                    self.test_data["package_x"].append(eval_infos[i][0]["package_x"])
                    self.test_data["v_deviation"].append(
                        eval_infos[i][0]["v_deviation"]
                    )
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.eval_infos = eval_infos

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_average_steps": [np.mean(self.test_data["terminate_at"])],
            "eval_terminate_x": [np.mean(self.test_data["package_x"])],
            # "eval_v_deviation": self.test_data["v_deviation"],
            "eval_average_v_deviation": [np.mean(self.test_data["v_deviation"])],
        }
        # print(eval_env_infos)
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        # print(self.eval_episode_rewards)
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()
