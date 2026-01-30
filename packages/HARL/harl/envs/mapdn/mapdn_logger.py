from harl.common.base_logger import BaseLogger


class MAPDNLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(MAPDNLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False
        self.test_data = {
            "terminate_at": [],
            "percentage_of_v_out_of_control": [],
            "percentage_of_lower_than_lower_v": [],
            "percentage_of_higher_than_upper_v": [],
            "totally_controllable_ratio": [],
            "average_voltage_deviation": [],
            "average_voltage": [],
            "max_voltage_drop_deviation": [],
            "max_voltage_rise_deviation": [],
            "total_line_loss": [],
            "q_loss": [],
            "destroy": [],
            "sum_rewards": [],
        }
        self.episodes = -1

    def init(self, episodes):
        """Initialize the logger."""
        super().init(episodes)
        self.test_data = {
            "terminate_at": [],
            "percentage_of_v_out_of_control": [],
            "percentage_of_lower_than_lower_v": [],
            "percentage_of_higher_than_upper_v": [],
            "totally_controllable_ratio": [],
            "average_voltage_deviation": [],
            "average_voltage": [],
            "max_voltage_drop_deviation": [],
            "max_voltage_rise_deviation": [],
            "total_line_loss": [],
            "q_loss": [],
            "destroy": [],
            "sum_rewards": [],
        }
        if self.episodes != -1:
            self.episode_done = [False] * self.episodes

    def eval_init(self):
        super().eval_init()
        self.test_data = {
            "terminate_at": [],
            "percentage_of_v_out_of_control": [],
            "percentage_of_lower_than_lower_v": [],
            "percentage_of_higher_than_upper_v": [],
            "totally_controllable_ratio": [],
            "average_voltage_deviation": [],
            "average_voltage": [],
            "max_voltage_drop_deviation": [],
            "max_voltage_rise_deviation": [],
            "total_line_loss": [],
            "q_loss": [],
            "destroy": [],
            "sum_rewards": [],
        }
        if self.episodes != -1:
            self.episode_done = [False] * self.episodes

    def get_task_name(self):
        return "mapdn"

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        if self.episodes == -1:
            self.episodes = len(eval_infos)
            self.episode_done = [False] * self.episodes
        for i in range(len(eval_infos)):
            if eval_dones[i][0] and not self.episode_done[i]:
                self.episode_done[i] = True
                self.test_data["terminate_at"].append(eval_infos[i][0]["curr_step"])
                self.test_data["percentage_of_v_out_of_control"].append(
                    eval_infos[i][0]["percentage_of_v_out_of_control"]
                )
                self.test_data["percentage_of_lower_than_lower_v"].append(
                    eval_infos[i][0]["percentage_of_lower_than_lower_v"]
                )
                self.test_data["percentage_of_higher_than_upper_v"].append(
                    eval_infos[i][0]["percentage_of_higher_than_upper_v"]
                )
                self.test_data["totally_controllable_ratio"].append(
                    eval_infos[i][0]["totally_controllable_ratio"]
                )
                self.test_data["average_voltage_deviation"].append(
                    eval_infos[i][0]["average_voltage_deviation"]
                )
                self.test_data["average_voltage"].append(
                    eval_infos[i][0]["average_voltage"]
                )
                self.test_data["max_voltage_drop_deviation"].append(
                    eval_infos[i][0]["max_voltage_drop_deviation"]
                )
                self.test_data["max_voltage_rise_deviation"].append(
                    eval_infos[i][0]["max_voltage_rise_deviation"]
                )
                self.test_data["total_line_loss"].append(
                    eval_infos[i][0]["total_line_loss"]
                )
                self.test_data["q_loss"].append(eval_infos[i][0]["q_loss"])
                self.test_data["destroy"].append(eval_infos[i][0]["destroy"])
                self.test_data["sum_rewards"].append(eval_infos[i][0]["sum_rewards"])
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    # def eval_thread_done(self, eval_i):
    # super().eval_thread_done(eval_i)

    # def eval_log(self, eval_episode):
    #     """Log evaluation information at the end of an episode."""
    #     self.test_data["terminate_at"].append(eval_episode["terminate_at"])
