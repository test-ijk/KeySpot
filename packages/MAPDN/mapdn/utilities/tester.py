import torch as th
from mapdn.utilities.util import translate_action, prep_obs
import numpy as np
import time
import rich
import wandb
import copy


class PGTester(object):
    def __init__(self, args, behaviour_net, env, render=False):
        self.env = env
        self.behaviour_net = (
            behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
        )
        self.args = args
        self.device = th.device(
            "cuda" if th.cuda.is_available() and self.args.cuda else "cpu"
        )
        self.n_ = self.args.agent_num
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.render = render

    def run(self, day, hour, quarter):
        # reset env
        state, global_state = self.env.manual_reset(day, hour, quarter)

        # init hidden states
        last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

        record = {
            "pv_active": [],
            "pv_reactive": [],
            "bus_active": [],
            "bus_reactive": [],
            "bus_voltage": [],
            "line_loss": [],
        }

        record["pv_active"].append(self.env._get_sgen_active())
        record["pv_reactive"].append(self.env._get_sgen_reactive())
        record["bus_active"].append(self.env._get_res_bus_active())
        record["bus_reactive"].append(self.env._get_res_bus_reactive())
        record["bus_voltage"].append(self.env._get_res_bus_v())
        record["line_loss"].append(self.env._get_res_line_loss())

        for t in range(self.args.max_steps):
            if self.render:
                self.env.render()
                time.sleep(0.01)
            state_ = (
                prep_obs(state)
                .contiguous()
                .view(1, self.n_, self.obs_dim)
                .to(self.device)
            )
            action, _, _, _, hid = self.behaviour_net.get_actions(
                state_,
                status="test",
                exploration=False,
                actions_avail=th.tensor(self.env.get_avail_actions()),
                target=False,
                last_hid=last_hid,
            )
            _, actual = translate_action(self.args, action, self.env)
            reward, done, info = self.env.step(actual, add_noise=False)
            done_ = done or t == self.args.max_steps - 1
            record["pv_active"].append(self.env._get_sgen_active())
            record["pv_reactive"].append(self.env._get_sgen_reactive())
            record["bus_active"].append(self.env._get_res_bus_active())
            record["bus_reactive"].append(self.env._get_res_bus_reactive())
            record["bus_voltage"].append(self.env._get_res_bus_v())
            record["line_loss"].append(self.env._get_res_line_loss())
            next_state = self.env.get_obs()
            # set the next state
            state = next_state
            # set the next last_hid
            last_hid = hid
            if done_:
                break
        return record

    def _single_episode(self, seed=None):
        """
        单独运行一集，返回info和终止步数。
        """
        # 可选：设置随机种子，保证每个进程独立
        if seed is not None:
            np.random.seed(seed)
            import random

            random.seed(seed)
            import torch

            torch.manual_seed(seed)
        state, global_state = self.env.reset()
        last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
        info = None
        infos = []
        t_terminate = self.args.max_steps - 1
        for t in range(self.args.max_steps):
            if self.render:
                self.env.render()
                time.sleep(0.01)
            state_ = (
                prep_obs(state)
                .contiguous()
                .view(1, self.n_, self.obs_dim)
                .to(self.device)
            )
            action, _, _, _, hid = self.behaviour_net.get_actions(
                state_,
                status="test",
                exploration=False,
                actions_avail=th.tensor(self.env.get_avail_actions()),
                target=False,
                last_hid=last_hid,
            )
            _, actual = translate_action(self.args, action, self.env)
            reward, done, info = self.env.step(actual, add_noise=False)
            done_ = done or t == self.args.max_steps - 1
            next_state = self.env.get_obs()
            state = next_state
            last_hid = hid
            infos.append(copy.deepcopy(info))
            if done_:
                t_terminate = t
                break
        return infos, t_terminate

    def batch_run(self, num_epsiodes=100, gpus=[0, 1]):
        """
        使用joblib并行运行多个episode，统计所有episode的均值。
        """
        from joblib import Parallel, delayed
        import time

        start_time = time.time()

        def run_one(seed):
            env = copy.deepcopy(self.env)
            # 使用对应GPU上的模型

            args = copy.deepcopy(self.args)
            tester = PGTester(args, self.behaviour_net, env, self.render)

            result = tester._single_episode(seed)

            # 只释放环境和测试器
            del env
            del tester

            return result

        n_jobs = min(16, num_epsiodes)
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(seed) for seed in range(num_epsiodes)
        )

        elapsed = time.time() - start_time
        # 聚合统计
        test_results = {}
        terminate_cnt = 0
        terminate_steps = []
        for infos, t_terminate in results:
            for info in infos:
                for k, v in info.items():
                    if k == "sum_rewards":
                        continue
                    if "mean_" + k not in test_results:
                        test_results["mean_" + k] = [v]
                    else:
                        test_results["mean_" + k].append(v)
            if t_terminate < self.args.max_steps - 2:
                terminate_cnt += 1
                terminate_steps.append(t_terminate)
            if "mean_sum_rewards" not in test_results:
                test_results["mean_sum_rewards"] = [info["sum_rewards"]]
            else:
                test_results["mean_sum_rewards"].append(info["sum_rewards"])
        test_results["terminate_cnt"] = terminate_cnt
        test_results["mean_terminate_at_step"] = (
            np.mean(terminate_steps) if terminate_steps else 0
        )
        for k, v in test_results.items():
            if isinstance(v, list):
                test_results[k] = np.mean(v)
        rich.print(test_results)
        wandb.log(test_results)
        # 速度诊断
        avg_time = elapsed / num_epsiodes if num_epsiodes > 0 else 0
        print(
            f"[速度诊断] 总用时: {elapsed:.2f} 秒, 单个episode平均用时: {avg_time:.2f} 秒"
        )
        return test_results

    def print_info(self, stat):
        string = ["Test Results:"]
        for k, v in stat.items():
            string.append(k + f": mean: {v[0]:2.4f}, \t2std: {v[1]:2.4f}")
        string = "\n".join(string)
        print(string)
