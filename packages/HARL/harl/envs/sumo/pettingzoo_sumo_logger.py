from harl.common.base_logger import BaseLogger
import numpy as np


def get_default_test_data():
    return {
        "system_total_waiting_time": [],
        "tw_bigger_than_1000": [],
        "system_total_stopped": [],
        "system_step": [],
        
    }


class PettingZooSumoLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooSumoLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False
        self.test_data = get_default_test_data()

    def init(self, episodes):
        """Initialize the logger."""
        super().init(episodes)
        self.test_data = get_default_test_data()

    def get_task_name(self):
        return "sumo"

    def eval_init(self):
        super().eval_init()
        self.test_data = get_default_test_data()

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        super().eval_per_step(eval_data)
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        mean_value = sum(thread[0]["system_total_stopped"] for thread in eval_infos if thread) / len(eval_infos)
        a = [thread[0]["system_total_stopped"] for thread in eval_infos if thread]
        print(a)
        self.test_data["system_total_stopped"].append(mean_value)
        curr_step = [thread[0]["curr_step"] for thread in eval_infos if thread][0]
        
        # print(f"curr_step:{curr_step}")
        self.test_data['system_step'].append(curr_step)
        for i in range(len(eval_infos)):
            # print(f"len(eval_infos):{len(eval_infos)}")
            if eval_dones[i][0]:
                self.test_data["system_total_waiting_time"].append(
                    eval_infos[i][0]["system_total_waiting_time"]
                )
            if eval_infos[i][0]["system_total_waiting_time"] > 2000:
                self.test_data["tw_bigger_than_1000"].append(
                    eval_infos[i][0]["system_total_waiting_time"]
                )


    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_system_total_waiting_time_avg": [
                np.mean(self.test_data["system_total_waiting_time"])
            ],
            "eval_system_total_waiting_time_all": [
                np.sum(self.test_data["system_total_waiting_time"])
            ],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        # print(self.eval_episode_rewards)
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()
