from __future__ import annotations

import json
import os
import time

import hydra
import rich
from omegaconf import DictConfig
from rich.panel import Panel
import omegaconf
from copy import deepcopy
import wandb
import numpy as np
import matplotlib.pyplot as plt

from harl.runners import RUNNER_REGISTRY
from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
import hydra_type
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from moviepy.editor import VideoFileClip
import imageio

os.environ["SDL_VIDEODRIVER"] = "dummy"

# 设置字体路径
# font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 确保路径正确
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
wandb_results = []
max_cycles = 10000


def export_gif(
    config_name, frames_arr, rewards_arr, globalConfig: hydra_type.EntrypointConfig
):
    # 文件夹

    rich.print(f"Exporting gif for {config_name}")
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos/"
    )
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):
        # 1. gif生成
        rewards = rewards_arr[i]
        is_negative = rewards < 0  # {'fail' if is_negative else 'success'}_
        gif_path = os.path.join(
            gif_folder,
            f"{config_name}_[{rewards:.2f}]_{i}.gif",
        )
        imageio.mimwrite(
            gif_path,
            frames,
            duration=10,
        )

        # 3. 视频生成
        clip = VideoFileClip(gif_path)
        clip.write_videofile(
            os.path.join(
                gif_folder,
                f"{config_name}_[{rewards:.2f}]_{i}.mp4",
            ),
            codec="libx264",
            logger=None,
        )

        # 删除前面的gif
        os.remove(gif_path)


def _to_dict(cfg1: DictConfig):
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)


def log_wandb(config):
    print("logging to wandb")
    rich.print(wandb_results)
    angle_intervals = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 5),
        (5, 8),
        (8, 10),
        (10, 15),
        (15, float("inf")),
    ]
    columns = [
        "algo",
        "variant",
        "scenario",
        "n_walkers",
        "terminate_cnt",
        "package_final_x",
        "angle_avg",
    ]
    columns.extend([f"angle-{start}-{end}" for start, end in angle_intervals])
    columns.extend(
        [k for k, v in _to_dict(config)["env_tweak"].items() if not k.startswith("_")]
    )
    # 将字典数据转换为列表格式
    table_data = []
    for result in wandb_results:
        table_data.append(
            [
                result["algo"],
                result["variant"],
                result["scenario"],
                result["n_walkers"],
                result["terminate_cnt"],
                result["package_x"],
                result["angle_avg"],
                *[
                    result["angle_data"].get(f"angle-{start}-{end}", 0)
                    for start, end in angle_intervals
                ],
                *[
                    v
                    for k, v in _to_dict(config)["env_tweak"].items()
                    if not k.startswith("_")
                ],
            ]
        )

    rich.print(table_data)
    test_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"test_table": test_table})


def run_evaluations(
    config: hydra_type.EntrypointConfig, algorithm: str
) -> tuple[dict, dict]:
    """执行baseline和扰动测试的评估"""
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)
    results = []

    # 读取所有的scenarios，都要做对照实验
    for scenario_name, scenario in config.disturbances.items():
        scenario: hydra_type.ScenarioConfig = scenario
        # 这里容易搞混；请记住，下面的四个eval是对应同一个算法的四个变种。
        # 因此，面对某个扰动的环境，有四个checkpoint会被测试：在原环境训练、在angle环境训练、没有obs、有obs
        # 当环境定义为raw的时候，只测试前两个变种
        # raw, angle
        # + !scenario.is_raw -> obs, no_obs
        print("!!!!!!!!!!!!!!!!!!")
        print(scenario)
        raw_results = eval(
            config,
            algorithm=algorithm,
            checkpoint_type="raw",
            eval_scenario=scenario,
        )
        results.append(raw_results)

        # angle_results = eval(
        #     config,
        #     checkpoint=checkpoint,
        #     checkpoint_type="angle",
        #     eval_scenario=scenario,
        # )
        # results.append(angle_results)

        if not scenario.is_raw:
            obs_results = eval(
                config,
                algorithm=algorithm,
                checkpoint_type=f"disturb_{scenario.name}_obs",
                eval_scenario=scenario,
            )
            results.append(obs_results)
            if config.basic_config.ablation:
                no_obs_results = eval(
                    config,
                    algorithm=algorithm,
                    checkpoint_type=f"disturb_{scenario.name}_no_obs",
                    eval_scenario=scenario,
                )
                results.append(no_obs_results)

                # 以下是baseline或者说消融；把obs的checkpoint在没有扰动的环境上测试
                _sc = deepcopy(scenario)
                _sc.is_raw = True
                _sc.name = "raw"
                _sc.disturbances = None
                obs_raw_results = eval(
                    config,
                    algorithm=algorithm,
                    checkpoint_type=f"disturb_{scenario.name}_obs",
                    eval_scenario=_sc,
                )
                results.append(obs_raw_results)
                no_obs_raw_results = eval(
                    config,
                    algorithm=algorithm,
                    checkpoint_type=f"disturb_{scenario.name}_no_obs",
                    eval_scenario=_sc,
                )
                results.append(no_obs_raw_results)

    return results


def eval(
    globalConfig: hydra_type.EntrypointConfig,
    algorithm: str,
    checkpoint_type: str,
    eval_scenario: hydra_type.ScenarioConfig,
):
    start_time = time.time()
    base_checkpoint_path = f"./results/models/{globalConfig.basic_config.save_group}/pettingzoo_mw/multiwalker/{algorithm}/[{algorithm}]<{checkpoint_type}>"
    # if globalConfig.env_tweak.n_walkers != 3:
    for key in ["n_walkers", *globalConfig.env_tweak.tweak_types]:
        if not key.startswith("_"):
            base_checkpoint_path += f"<{key}={globalConfig.env_tweak[key]}>"
    rich.print(os.listdir(base_checkpoint_path))
    seed_folder = next(
        folder
        for folder in os.listdir(base_checkpoint_path)
        if folder.startswith("seed-")
    )
    checkpoint_path = os.path.join(base_checkpoint_path, seed_folder, "models")
    print(checkpoint_type)
    # 1. 先读取对应的模型
    rich.print(
        Panel(
            f"Checkpoint Path: {checkpoint_path}\nScenario Name: {eval_scenario.name}",
            title="Evaluation Info",
        )
    )

    # 1.1. 从配置里读取参数，转换为harl使用的格式
    with initialize(version_base=None, config_path="./configs"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"algorithm={algorithm}",
                f"environment={checkpoint_type}",
            ],
        )
        algo_args = cfg.algorithm
        env_args = cfg.environment

        algorithm_name = cfg.algorithm.name
        env_name = cfg.environment.name
        scenario_name = cfg.environment.scenario
        basic_info = {
            "env": env_name,
            "algo": algorithm_name,
            "exp_name": f"testing_<{algorithm_name}>_{scenario_name}",
        }
        # 特殊处理max_cycles
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("episode_length") is not None
        ):
            algo_args.train.episode_length = globalConfig.env_tweak.max_cycles
        env_args.max_cycles = globalConfig.env_tweak.max_cycles

        algo_args.train.model_dir = checkpoint_path  # 读取模型！

        # 配置eval遍数
        algo_args.eval.n_eval_rollout_threads = globalConfig.basic_config.eval_threads
        algo_args.eval.eval_episodes = globalConfig.basic_config.eval_episodes

        # gpu
        algo_args.device.cuda = globalConfig.basic_config.use_gpu

        # 配置render
        if globalConfig.basic_config.render:
            algo_args.render.use_render = True
            algo_args.render.render_episodes = globalConfig.basic_config.eval_episodes

        # 配置num_env_steps
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1  # FIXME: ???

        algo_dict = _to_dict(algo_args)
        del algo_dict["name"]

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]

        # 处理env_tweaks
        for key, value in _to_dict(globalConfig.env_tweak).items():
            if value is not None:
                env_dict[key] = value
        env_dict["custom"]["eval_disturb"] = _to_dict(eval_scenario)["disturbances"]
        env_dict["custom"]["is_eval"] = True
        del env_dict["tweak_types"]

        # ============== 处理完毕config，启动runner实例 ==============
        runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

        if globalConfig.basic_config.render:
            render_mode = "rgb_array"
            rgb_array, rewards_arr, episode_obses_arr, lidar_obs_arr = runner.render(
                render_mode
            )
            config_name = f"[{algorithm}]<{checkpoint_type}>_{eval_scenario.name}"
            for key in ["n_walkers", *globalConfig.env_tweak.tweak_types]:
                if not key.startswith("_"):
                    config_name += f"<{key}={globalConfig.env_tweak[key]}>"
            # 保存episode_obses_arr到JSON文件
            if rgb_array is not None and episode_obses_arr is not None:
                json_dir = os.path.join(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    "./data/",
                )
                os.makedirs(json_dir, exist_ok=True)

                json_path = os.path.join(json_dir, f"{config_name}_episode_obses.json")

                # # 将numpy数组转换为列表以便JSON序列化
                episode_obses_serializable = []
                for episode in episode_obses_arr:
                    episode_serializable = []
                    for agent_obses in episode:
                        episode_serializable.append(
                            [obs.tolist() for obs in agent_obses]
                        )
                    episode_obses_serializable.append(episode_serializable)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(episode_obses_serializable, f, ensure_ascii=False)

                # 和上面一样，也存一份lidar_obs_arr
                json_path = os.path.join(json_dir, f"{config_name}_lidar_obs.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(lidar_obs_arr, f, ensure_ascii=False)

                rich.print(f"Episode observations saved to: {json_path}")

            export_gif(
                config_name=config_name,
                frames_arr=rgb_array,
                rewards_arr=rewards_arr,
                globalConfig=globalConfig,
            )

            exit()
        else:
            # 根据是否是off-policy，选择不同的eval方式
            has_logger = hasattr(runner, "logger")
            if has_logger:
                logger: PettingZooMWLogger = runner.logger
                logger.is_testing = (
                    True  # 标识目前在eval；但是eval这个词被它用了，只能用test了。
                )
                runner.eval()
                terminate_arr = logger.test_data["terminate_at"]
                angle_arr = logger.test_data["angle_data"]
            else:
                logger = None
                runner.eval(1)
                terminate_arr = runner.eval_episode_lens
                angle_arr = runner.eval_episode_angles

            # 开始计算
            # 2.1 计算提前摔倒的次数
            terminate_cnt = 0
            package_x = []
            for i in range(len(terminate_arr)):
                if (
                    terminate_arr[i] + 2 < globalConfig.env_tweak.max_cycles
                ):  # +2 去除一点边际问题
                    terminate_cnt += 1
                package_x.append(
                    logger.test_data["package_x"][i]
                    if has_logger
                    else runner.episode_xs[i]
                )
            # 关闭eval_envs和runner
            if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
                runner.eval_envs.close()
            runner.close()

            end_time = time.time()
            print(
                f"处理[{algorithm}]<{checkpoint_type}>_{eval_scenario.name} 耗时: {end_time - start_time:.2f}秒"
            )
            return_result = {
                "desc": f"[{algorithm}]<{checkpoint_type}>_{eval_scenario.name}_{_to_dict(eval_scenario).get('desc', 'original')}",
                "algo": algorithm,
                "variant": checkpoint_type,
                "scenario": eval_scenario.name,
                "terminate_cnt": terminate_cnt,
                "angle_data": [
                    angle for episode_angles in angle_arr for angle in episode_angles
                ],
                "angle_data_grouped": angle_arr,
                "package_x": sum(package_x) / len(package_x),
            }
            return return_result


def save_eval_results(results: dict, output_dir: str, name: str) -> str:
    """保存评估结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f)
    return filepath


def load_eval_results(filepath: str) -> dict:
    """从JSON文件加载评估结果,并保存副本到hydra输出目录"""
    with open(filepath, "r") as f:
        data = json.load(f)

    return data


def analyze_eval_results(
    results,
    config: hydra_type.EntrypointConfig,
    algorithm: str,
):
    for res in results:
        angle_intervals = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 10),
            (10, 15),
            (15, float("inf")),
        ]
        baseline_interval_counts = {
            f"angle-{start}-{end}": sum(
                1 for a in res["angle_data"] if start <= abs(a) < end
            )
            for start, end in angle_intervals
        }
        angle_avg = sum(abs(a) for a in res["angle_data"]) / len(res["angle_data"])

        # 绘制折线图（每组单独一张）
        if config.cherry_pick.export_angle_data:
            angle_groups = res["angle_data_grouped"]
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            save_dir = os.path.join(output_dir, "angle_figs")
            os.makedirs(save_dir, exist_ok=True)
            for idx, group in enumerate(angle_groups):
                x = np.arange(len(group))
                y = np.abs(group)
                plt.figure(figsize=(12, 6))
                plt.plot(x, y, label="|angle|", linewidth=2)
                plt.xlabel("Step", fontsize=36)
                plt.ylabel("Angle (abs)", fontsize=36)
                plt.grid(True)
                plt.legend(fontsize=36)
                plt.ylim(0, 10)
                plt.xlim(0, 500)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                fname = f"{res['scenario']}_cp_{res['variant']}_{idx}.png"
                fname = fname.replace("/", "_")  # 防止路径问题
                plt.savefig(os.path.join(save_dir, fname))
                plt.close()

        wandb_item = {
            "algo": res["algo"],
            "variant": res["variant"],
            "scenario": res["scenario"],
            "terminate_cnt": res["terminate_cnt"],
            "package_x": res["package_x"],
            "angle_data": baseline_interval_counts,
            "angle_avg": angle_avg,
            **{
                k: v
                for k, v in _to_dict(config)["env_tweak"].items()
                if not k.startswith("_")
            },
        }
        rich.print(wandb_item)
        wandb_results.append(wandb_item)

    rich.print(wandb_results)


@hydra.main(
    config_path="./configs/evaluation",
    config_name="eval",
    version_base=None,
)
def main(cfg: hydra_type.EntrypointConfig):
    # 用于json存储的目录
    json_dir = "./results/eval_results"
    os.makedirs(json_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H:%M")
    GlobalHydra.instance().clear()

    # 初始化wandb
    run = wandb.init(
        project=cfg.basic_config.wandb_project,
        name=cfg.algorithm + "_" + timestamp,
        config=_to_dict(cfg),
        save_code=True,
        group=cfg.basic_config.run_group,
        job_type="eval" if not cfg.basic_config.render else "render",
    )

    def process_checkpoint(algorithm: str):
        print(f"Processing checkpoint: {algorithm}")
        should_load_results = cfg.basic_config.load_results
        if should_load_results:
            # 加载已有结果模式
            result_file_name = cfg.basic_config.result_file_name
            if result_file_name == "latest":
                # 从latest子目录加载latest版本
                latest_dir = os.path.join(json_dir, "latest")
                results = load_eval_results(
                    os.path.join(latest_dir, f"{algorithm}.json")
                )
            else:
                # 从时间戳子目录加载指定版本
                timestamp_dir = os.path.join(json_dir, result_file_name)
                results = load_eval_results(
                    os.path.join(timestamp_dir, f"{algorithm}.json")
                )
        else:
            # 执行评估模式
            results = run_evaluations(cfg, algorithm)

            # 创建时间戳子目录
            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            # 创建latest子目录
            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            # 保存结果到时间戳子目录
            save_eval_results(
                results,
                timestamp_dir,
                f"{algorithm}",
            )

            save_eval_results(results, latest_dir, f"{algorithm}")

        # 分析结果并获取图表
        wandb_results = []
        rich.print(wandb_results)
        analyze_eval_results(results, cfg, algorithm)
        log_wandb(cfg)

    start_time = time.time()
    process_checkpoint(cfg.algorithm)
    end_time = time.time()
    print(f"{cfg.algorithm} 耗时: {end_time - start_time:.2f}秒")

    # 发送完成通知
    # requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/Eval完成")
    run.finish()
    # exit()


if __name__ == "__main__":
    main()
