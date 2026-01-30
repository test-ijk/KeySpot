from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
import rich.pretty
import time
import wandb
import hydra
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime
from .types.task.eval_type import EvalConfig
from .train import _to_dict
from .train import _to_harl_dict as _train_to_harl_dict
import os
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
import json
import atexit
from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio
from typing import cast
from enum import Enum
import numpy as np
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"


class Env(Enum):
    MAPDN = "mapdn"
    SUMO = "sumo"
    PETTINGZOO_MW = "pettingzoo_mw"
    PETTINGZOO_MW_LLM = "pettingzoo_mw_llm"
    SUMO_LLM = "sumo_llm"


def _to_harl_dict(
    cfg: EvalConfig,
):
    
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _train_to_harl_dict(cfg)

    rich.print(algo_dict)
    # sys.exit()

    if (
        hasattr(cfg.eval_scenario, "env_tweak")
        and cfg.eval_scenario.env_tweak is not None
    ):
        eval_env_tweak = _to_dict(cfg.eval_scenario.env_tweak)
        for key in eval_env_tweak.keys():
            if not key.startswith("_") and key != "tweak_types":
                env_dict[key] = eval_env_tweak[key]
                print(f"eval_env_tweak: {key} = {eval_env_tweak[key]}")

    if hasattr(cfg.eval_scenario, "events") and cfg.eval_scenario.events is not None:
        env_dict["events"] = _to_dict(cfg.eval_scenario)["events"]
    

    return (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    )


def export_gif(config_name, frames_arr, rewards_arr):
    # 文件夹

    rich.print(f"Exporting gif for {config_name}")
    
    # 尝试获取Hydra输出目录，如果失败则使用默认目录
    try:
        base_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        gif_dir = os.path.join(base_dir, "./videos/")
    except ValueError:
        # HydraConfig未设置（使用hydra.compose时），使用默认目录
        gif_dir = "./results/renders/"
    
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):
        # 1. gif生成
        rewards = rewards_arr[i]
        gif_path = os.path.join(
            gif_folder,
            f"{config_name}_[{rewards:.2f}]_{i}.gif",
        )
        imageio.mimwrite(
            gif_path,
            frames,
            fps=10,
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


def eval(
    config: EvalConfig,
):
    rich.print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 0. 处理参数
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _to_harl_dict(config)

    this_env = Env(env_name)
    this_env_is_mw_series = (
        this_env == Env.PETTINGZOO_MW or this_env == Env.PETTINGZOO_MW_LLM
    )

    # 1. 加载模型
    env_folder = ""
    if this_env == Env.PETTINGZOO_MW:
        env_folder = "multiwalker"
    elif this_env == Env.PETTINGZOO_MW_LLM:
        env_folder = "multiwalker"
    elif this_env == Env.SUMO:
        env_folder = "sumo"
    elif this_env == Env.SUMO_LLM:
        env_folder = "sumo"
    elif this_env == Env.MAPDN:
        env_folder = "mapdn"
    model_path = f"./results/models/{save_group}/{env_name}/{env_folder}/{algorithm_name}/[{algorithm_name}]<{scenario_name}>"
    rich.print(f"Loading model from {model_path}")

    # 2. 通用的env_tweak方法
    name_suffix = ""
    rich.print(config.environment.env_tweak.tweak_types)
    tweak_types = config.environment.env_tweak.tweak_types
    if this_env_is_mw_series:
        tweak_types = ["n_walkers", *sorted(config.environment.env_tweak.tweak_types)]
    env_tweaks = _to_dict(config.environment.env_tweak)
    for key in tweak_types:
        if not key.startswith("_"):
            name_suffix += f"<{key}={env_tweaks.get(key, None)}>"
    model_path += name_suffix

    seed_folder = next(
        folder for folder in os.listdir(model_path) if folder.startswith("seed-")
    )

    checkpoint_path = os.path.join(model_path, seed_folder, "models")
    rich.print(f"Loading model from {checkpoint_path}")

    # 2. 修改参数为eval可用的
    def _modify_algo_and_env_dict():
        algo_dict["train"]["model_dir"] = checkpoint_path  # 模型位置

        # eval thread
        algo_dict["eval"]["n_eval_rollout_threads"] = (
            config.eval_settings.general.eval_threads
        )
        algo_dict["eval"]["eval_episodes"] = config.eval_settings.general.eval_episodes

        # render
        algo_dict["render"]["use_render"] = config.eval_settings.functions.render
        algo_dict["render"]["render_episodes"] = (
            config.eval_settings.functions.render_episodes
        )

        # logger
        algo_dict["logger"]["log_dir"] = f"./results/logs/{save_group}"

        # FIXME: 为什么需要这个？
        if (this_env_is_mw_series) and algo_dict["train"].get(
            "num_env_steps"
        ) is not None:
            algo_dict["train"]["num_env_steps"] = 1  # FIXME: ???

        # disturbances的引入
        if this_env_is_mw_series:
            env_dict["custom"]["is_eval"] = True
            env_dict["custom"]["eval_disturb"] = _to_dict(config.eval_scenario).get(
                "disturbances", []
            )

    _modify_algo_and_env_dict()

    # rich.pretty.pprint(algo_dict, expand_all=True)
    # rich.pretty.pprint(env_dict, expand_all=True)

    # 3. 初始化runner
    print("初始化runner")
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)
    print("there!!!!!!!!!!!!!!!!!!!!!!!")

    # 修改：
    try:
        print("\n--- Listing Environment Object Attributes ---")
    
        # 打印 runner.envs 对象的所有属性和方法
        env_attributes = dir(runner.envs)
        rich.print(env_attributes)

        print("---------------------------\n")
                
      
    except Exception as e:
        print(f"Failed to print agents due to an error: {e}")
    
    # sys.stdout.flush()
    # sys.exit()
    # 修改结束

    @atexit.register
    def _cleanup():
        runner.close()
        # wandb.finish()

    is_online_policy = hasattr(runner, "logger")
    start_time = time.time()
    # 4. render？还是eval？
    print("参数拿到了")
    if config.eval_settings.functions.render:

        def _render():
            render_mode = "rgb_array"
            if is_online_policy:
                (
                    rgb_array,
                    rewards_arr,
                    episode_obses_arr,
                    lidar_obs_arr,
                ) = runner.render(render_mode)
            else:
                (
                    rgb_array,
                    rewards_arr,_,_
                ) = runner.render(render_mode)
            config_name = f"[{algorithm_name}]<{env_name}>_{scenario_name}{name_suffix}"
            if rgb_array is not None:
                export_gif(
                    config_name=config_name,
                    frames_arr=rgb_array,
                    rewards_arr=rewards_arr,
                )
            if not is_online_policy:
                return
            # 保存episode_obses_arr到JSON文件
            if (
                episode_obses_arr is not None
                and config.eval_settings.functions.export_angle_data
            ):
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

        _render()
        if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
            runner.eval_envs.close()
        runner.close()
        end_time = time.time()
        print(f"Render time: {end_time - start_time} seconds")
    else:
        # 根据是否是off-policy，选择不同的eval方式
        angle_arr = []
        package_contact_arr = []
        if is_online_policy:
            print("it is onpolicy")
            runner = cast(OnPolicyMARunner, runner)
            logger: PettingZooMWLogger = runner.logger
            logger.is_testing = (
                True  # 标识目前在eval；但是eval这个词被它用了，只能用test了。
            )
            runner.eval()
            assert runner.eval_envs is not None
            runner.eval_envs.reset()
            terminate_arr = logger.test_data.get("terminate_at", [])
            if this_env_is_mw_series:
                angle_arr = logger.test_data.get("angle_data", [])
                
                # 将线程级别数据转换为episode级别数据
                def convert_thread_data_to_episodes(thread_angle_data, terminate_arr, n_threads):
                    """
                    将线程级别的角度数据转换为episode级别
                    
                    Args:
                        thread_angle_data: list of list, thread_angle_data[tid] = 该线程的所有角度
                        terminate_arr: list, terminate_arr[i] = Episode i 的终止步数
                        n_threads: int, 并行线程数
                    
                    Returns:
                        episode_angles: list of list, episode_angles[i] = Episode i 的角度数据
                    """
                    total_episodes = len(terminate_arr)
                    episode_angles = []
                    
                    # 每个线程处理的episodes
                    thread_cursors = [0] * n_threads  # 每个线程当前读取到的位置
                    
                    for ep_idx in range(total_episodes):
                        thread_id = ep_idx % n_threads  # Episode被分配到哪个线程
                        start = thread_cursors[thread_id]
                        end = start + terminate_arr[ep_idx]
                        
                        # 从对应线程提取这个episode的数据
                        if thread_id < len(thread_angle_data) and end <= len(thread_angle_data[thread_id]):
                            ep_angles = thread_angle_data[thread_id][start:end]
                            episode_angles.append(ep_angles)
                            thread_cursors[thread_id] = end
                        else:
                            # 数据不足，添加空列表
                            episode_angles.append([])
                    
                    return episode_angles
                
                # 应用转换：将线程级别数据转换为episode级别
                n_threads = logger.algo_args["eval"]["n_eval_rollout_threads"]
                angle_arr = convert_thread_data_to_episodes(angle_arr, terminate_arr, n_threads)
                
                # 获取package接触地面的数据
                package_contact_arr = []
                try:
                    # Method 1: Get from logger.package_contact_history (collected during eval)
                    if hasattr(logger, 'package_contact_history'):
                        package_contact_arr = logger.package_contact_history
                        print(f"[DEBUG] Collected {len(package_contact_arr)} package contact records from logger")
                    else:
                        # Method 2: Try to access environment directly (for DummyVecEnv)
                        try:
                            env_instance = runner.eval_envs.env
                            if hasattr(env_instance, 'get_all_package_contact_history'):
                                package_contact_arr = env_instance.get_all_package_contact_history()
                                print(f"[DEBUG] Collected {len(package_contact_arr)} package contact records from environment")
                        except (AttributeError, Exception):
                            print(f"[DEBUG] No package contact history available")
                            package_contact_arr = []
                except Exception as e:
                    print(f"Warning: Failed to get package contact history: {e}")
                    import traceback
                    traceback.print_exc()
                    package_contact_arr = []

        else:
            print("it is offpolicy")
            runner = cast(OffPolicyBaseRunner, runner)
            logger = None
            runner.eval(1)
            terminate_arr = runner.eval_episode_lens
            angle_arr = runner.eval_episode_angles

        # 开始计算
        # 2.1 计算提前摔倒的次数
        terminate_cnt = 0
        package_x = []
        early_terminate_arr = []
        for i in range(len(terminate_arr)):
            if (
                terminate_arr[i] + 2 < config.environment.env_tweak.max_cycles
            ):  # +2 去除一点边际问题
                terminate_cnt += 1
                early_terminate_arr.append(terminate_arr[i])
            if this_env_is_mw_series and is_online_policy:
                package_x.append(
                    logger.test_data["package_x"][i]
                    if is_online_policy
                    else runner.episode_xs[i]  # type: ignore
                )
        # 关闭eval_envs和runner
        if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
            runner.eval_envs.close()
        runner.close()

        end_time = time.time()

        assert config.environment.env_tweak.max_cycles is not None

        if this_env_is_mw_series:
            
            
            angle_flatten = [
                angle for episode_angles in angle_arr for angle in episode_angles
            ]

            return_result = {
                "desc": f"[{algorithm_name}]<{scenario_name}>_{config.eval_scenario.name}_{_to_dict(config.eval_scenario).get('desc', 'original')}",
                "algo": algorithm_name,
                "variant": scenario_name,
                "scenario": config.eval_scenario.name,
                "terminate_cnt": terminate_cnt,
                "avg_terminate_at": sum(early_terminate_arr) / len(early_terminate_arr) if early_terminate_arr else 0,
                "total_episodes": len(terminate_arr),
                "total_time": end_time - start_time,
                "total_timesteps": sum(terminate_arr)
                + (config.eval_settings.general.eval_episodes - terminate_cnt)
                * config.environment.env_tweak.max_cycles,
                "total_timesteps1": sum(terminate_arr)
                + (config.eval_settings.general.eval_episodes - len(terminate_arr))
                * config.environment.env_tweak.max_cycles,
                "terminate_arr": terminate_arr,
                "total_timesteps1": sum(terminate_arr),
                # "angle_data": angle_flatten,
                "angle_data_grouped": angle_arr,
                "package_contact_arr": package_contact_arr,
            }
            
            # 添加扰动相关指标
            if hasattr(config.environment.env_tweak, 'disturbance_mode') and config.environment.env_tweak.disturbance_mode == 'adaptive':
                return_result["disturbance_config"] = {
                    "target_agent": getattr(config.environment.env_tweak, 'disturb_target_agent', None),
                    "magnitude": getattr(config.environment.env_tweak, 'disturb_magnitude', None)
                }
                
                if logger and hasattr(logger, 'test_data'):
                    mttf_values = [x for x in logger.test_data.get("mttf_data", []) if x is not None]
                    recovery_values = [x for x in logger.test_data.get("recovery_time_data", []) if x is not None]
                    max_angles = logger.test_data.get("max_angle_data", [])
                    
                    return_result["disturbance_mttf_avg"] = np.mean(mttf_values) if mttf_values else None
                    return_result["disturbance_recovery_time_avg"] = np.mean(recovery_values) if recovery_values else None
                    return_result["disturbance_max_angle"] = max(max_angles) if max_angles else None
            if is_online_policy:
                return_result["angle_data_avg"] = sum(angle_flatten) / len(
                    angle_flatten
                )
                return_result["angle_data_std"] = np.std(angle_flatten)
                return_result["angle_larger_than_5"] = sum(
                    [1 for angle in angle_flatten if angle > 5]
                ) / len(angle_flatten)
                return_result["angle_larger_than_10"] = sum(
                    [1 for angle in angle_flatten if angle > 10]
                ) / len(angle_flatten)
                return_result["angle_larger_than_15"] = sum(
                    [1 for angle in angle_flatten if angle > 15]
                ) / len(angle_flatten)
                return_result["package_x"] = sum(package_x) / len(package_x)

                results_dir = "/root/2507-multiwalker-harl/z_picture"
                disabled_agent_id = getattr(config.environment.env_tweak, 'disabled_walker_id', None)
                disturb = getattr(config.environment.env_tweak, 'disturb', None)
                
                if disabled_agent_id is not None and disturb is not None:
                    filename = f"agent_{disabled_agent_id}_disturb_'{disturb}'.txt"
                    filepath = os.path.join(results_dir, filename)
                else:
                    filepath = None
                
                if filepath and len(angle_flatten) > 0:
                    # 计算大于6.5的数据比例
                    threshold = 6.5
                    count_above_threshold = sum(1 for angle in angle_flatten if angle > threshold)
                    percentage_above = (count_above_threshold / len(angle_flatten)) * 100
                    
                    # 检查是否超过10%
                    # if percentage_above > 10:
                    # 输出文件
                    with open(filepath, 'w') as f:
                        f.write(f"数据统计报告\n")
                        f.write(f"==============\n")
                        f.write(f"总数据点: {len(angle_flatten)}\n")
                        f.write(f"阈值: {threshold}\n")
                        f.write(f"超过阈值的数据点: {count_above_threshold}\n")
                        f.write(f"比例: {percentage_above:.2f}%\n\n")
                        f.write("超过阈值的数据详情:\n")

        elif this_env == Env.MAPDN:
            return_result = {
                "desc": f"[{algorithm_name}]<{scenario_name}>_{config.eval_scenario.name}_{_to_dict(config.eval_scenario).get('desc', 'original')}",
                "algo": algorithm_name,
                "variant": scenario_name,
                "scenario": config.eval_scenario.name,
                "terminate_cnt": terminate_cnt,
                "avg_terminate_at": sum(early_terminate_arr) / len(early_terminate_arr) if early_terminate_arr else 0,
                "total_episodes": len(terminate_arr),
            }
            if is_online_policy:
                return_result["percentage_of_v_out_of_control"] = sum(
                    logger.test_data["percentage_of_v_out_of_control"]
                ) / len(logger.test_data["percentage_of_v_out_of_control"])
                return_result["percentage_of_lower_than_lower_v"] = sum(
                    logger.test_data["percentage_of_lower_than_lower_v"]
                ) / len(logger.test_data["percentage_of_lower_than_lower_v"])
                return_result["percentage_of_higher_than_upper_v"] = sum(
                    logger.test_data["percentage_of_higher_than_upper_v"]
                ) / len(logger.test_data["percentage_of_higher_than_upper_v"])
                return_result["totally_controllable_ratio"] = sum(
                    logger.test_data["totally_controllable_ratio"]
                ) / len(logger.test_data["totally_controllable_ratio"])
                return_result["average_voltage_deviation"] = sum(
                    logger.test_data["average_voltage_deviation"]
                ) / len(logger.test_data["average_voltage_deviation"])
                return_result["average_voltage"] = sum(
                    logger.test_data["average_voltage"]
                ) / len(logger.test_data["average_voltage"])
                return_result["max_voltage_drop_deviation"] = sum(
                    logger.test_data["max_voltage_drop_deviation"]
                ) / len(logger.test_data["max_voltage_drop_deviation"])
                return_result["max_voltage_rise_deviation"] = sum(
                    logger.test_data["max_voltage_rise_deviation"]
                ) / len(logger.test_data["max_voltage_rise_deviation"])
                return_result["total_line_loss"] = sum(
                    logger.test_data["total_line_loss"]
                ) / len(logger.test_data["total_line_loss"])
                return_result["q_loss"] = sum(logger.test_data["q_loss"]) / len(
                    logger.test_data["q_loss"]
                )
                return_result["destroy"] = sum(logger.test_data["destroy"]) / len(
                    logger.test_data["destroy"]
                )
                return_result["sum_rewards"] = sum(
                    logger.test_data["sum_rewards"]
                ) / len(logger.test_data["sum_rewards"])


        elif this_env == Env.SUMO or this_env == Env.SUMO_LLM:
            stopped_step_pairs = [
                {"step": step, "stopped": stopped} 
                for i, (step, stopped) in enumerate(zip(
                    logger.test_data["system_step"], 
                    logger.test_data["system_total_stopped"]
                ))
            ]
            return_result = {
                "desc": f"[{algorithm_name}]<{scenario_name}>_{config.eval_scenario.name}_{_to_dict(config.eval_scenario).get('desc', 'original')}",
                "algo": algorithm_name,
                "variant": scenario_name,
                "scenario": config.eval_scenario.name,
                "tw_bigger_than_1000": len(logger.test_data["tw_bigger_than_1000"]),
                "tw_bigger_than_1000_avg": sum(logger.test_data["tw_bigger_than_1000"])
                / (len(logger.test_data["tw_bigger_than_1000"]) + 1),
                "system_total_waiting_time": sum(
                    logger.test_data["system_total_waiting_time"]
                )
                / (len(logger.test_data["system_total_waiting_time"]) + 1),
                "system_total_stopped":stopped_step_pairs,
                # "system_step":logger.test_data["system_step"],
                "tw_bigger_than_1000_max": max(
                    logger.test_data["system_total_waiting_time"]
                ),
            }
            # if this_env == Env.SUMO_LLM:
            # return_result["llm_time"] = sum(logger.test_data["llm_time"])
            # / len(logger.test_data["llm_time"])
        else:
            return_result = {
                "desc": f"[{algorithm_name}]<{scenario_name}>_{config.eval_scenario.name}_{_to_dict(config.eval_scenario).get('desc', 'original')}",
                "algo": algorithm_name,
                "variant": scenario_name,
                "scenario": config.eval_scenario.name,
            }
        print(f"Evaluation time: {end_time - start_time} seconds")

        return_result["full_configs"] = _to_dict(config)
        return return_result

    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time} seconds")


def plot_angle_time_graph(angle_data, save_path, config_info):
    """绘制角度-时间变化图
    
    Args:
        angle_data: 角度数据列表的列表 [[ep1_angles], [ep2_angles], ...]
        save_path: 保存路径
        config_info: 配置信息字典
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, episode_angles in enumerate(angle_data):
        x = np.arange(len(episode_angles))
        y = np.array(episode_angles)
        color = color_cycle[i % len(color_cycle)]
        plt.plot(x, y, label=f'Episode {i+1}', color=color, alpha=0.7)
    
    # 绘制阈值线
    plt.axhline(y=6.5, color='red', linestyle='--', linewidth=2, label='Failure Threshold (6.5°)')
    plt.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, label='Recovery Threshold (5.0°)')
    
    # 标注扰动注入时间点
    if config_info.get('disturb_start_step'):
        plt.axvline(x=config_info['disturb_start_step'], color='green', 
                   linestyle=':', linewidth=2, label='Disturbance Injection')
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Package Angle (degrees)', fontsize=12)
    plt.title(f"Agent {config_info.get('target_agent', 'N/A')}, "
              f"Magnitude {config_info.get('magnitude', 'N/A')}", fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Angle-time plot saved to: {save_path}")


@hydra.main(
    config_path="../1.config/task/eval", config_name="default", version_base=None
)
def main(cfg: EvalConfig):
    # rich.pretty.pprint(_to_dict(cfg), expand_all=True)

    # 2. 整理参数，转换为dict以传导给harl
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _to_harl_dict(cfg)

    # 3. 初始化wandb
    wandb.init(
        project=cfg.wandb.wandb_project,
        config={"original": _to_dict(cfg), "algo": algo_dict, "env": env_dict},
        sync_tensorboard=True,
        # name=run_name + f"_{ts}",
        group=run_group,
        job_type="eval",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
        ],
    )

    import numpy as np

    def convert_np(obj):
        """
        递归地将dict/list中的numpy类型转换为Python原生类型
        """
        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # time.sleep(10000)

    # 5. 启动训练
    print("start!!!!!!!!!!!!!!!!!!!!!!!!")
    result = eval(cfg)
    # 保存结果到JSON文件
    """
    将评估结果保存为JSON格式文件
    根据当前时间创建目录结构并保存结果
    """
    if result is not None:
        import json
        from datetime import datetime

        # 获取当前时间
        now = datetime.now()
        mmdd = now.strftime("%m%d")
        hhmm = now.strftime("%H%M%S")

        # 创建保存路径
        save_dir = f"./results/runs/{mmdd}/{hhmm}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存为JSON文件
        """
        处理result中的numpy类型（如np.int32），将其转换为Python原生类型，确保可以被json序列化
        """
        

        result = convert_np(result)
        
        # 获取 perturb_targets 用于文件名
        if hasattr(cfg.environment.env_tweak, 'perturb_targets'):
            perturb_targets = cfg.environment.env_tweak.perturb_targets
            if perturb_targets and len(perturb_targets) > 0:
                place_suffix = "_".join(perturb_targets)
            else:
                place_suffix = "no_perturb"
            
            json_path = os.path.join(
                save_dir, f"{env_name}_{algorithm_name}_{scenario_name}_{place_suffix}.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            json_path = os.path.join(
                save_dir, f"{env_name}_{algorithm_name}_{scenario_name}.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {json_path}")

        wandb.log(result)


def filter_recovered_episodes(angle_data_grouped, terminate_arr, max_cycles=1000, package_contact_arr=None):
    """
    筛选"不稳定后恢复"的episodes
    
    过滤规则：
    1. 正常完成（terminate_arr[i] == max_cycles）
    2. Package未接触地面（新增，替代角度方差检查）
    3. 曾经失稳（max(abs(angles)) > 6.5°）
    4. 后来恢复（有步数 < 5.0°）
    
    Returns:
        recovered: list of recovered episode info
        episode_details: list of all episodes with filter reasons
    """
    recovered = []
    episode_details = []
    
    print(f"\n[FILTER] Checking {len(angle_data_grouped)} episodes...")
    
    for idx, (angles, term_step) in enumerate(zip(angle_data_grouped, terminate_arr)):
        print(f"\n[Episode {idx}]")
        
        episode_info = {
            'episode_idx': idx,
            'term_step': term_step,
            'status': 'unknown',
            'filter_reason': None,
            'max_angle': None,
            'std_angle': None,
            'failure_step': None,
            'recovery_step': None,
            'recovery_time': None,
            'first_unstable_step': None,
            'last_unstable_step': None,
            'unstable_duration': None
        }
        
        # 基本检查
        if not angles or len(angles) < 10:
            msg = f"Too few data points: {len(angles) if angles else 0}"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        angles_array = np.abs(np.array(angles))  # 使用绝对值
        max_angle = np.max(angles_array)
        std_angle = np.std(angles_array)
        episode_info['max_angle'] = float(max_angle)
        episode_info['std_angle'] = float(std_angle)
        
        # 计算第一次和最后一次失稳的步数（对所有episodes，包括失败的）
        first_unstable_step = None
        last_unstable_step = None
        for j, angle in enumerate(angles):
            if abs(angle) > 6.5:
                if first_unstable_step is None:
                    first_unstable_step = j
                last_unstable_step = j  # 持续更新，记录最后一次
        
        episode_info['first_unstable_step'] = first_unstable_step
        episode_info['last_unstable_step'] = last_unstable_step
        if first_unstable_step is not None and last_unstable_step is not None:
            episode_info['unstable_duration'] = last_unstable_step - first_unstable_step
        
        print(f"  Length: {len(angles)}, Term step: {term_step}")
        print(f"  Max angle: {max_angle:.2f}°, Std: {std_angle:.2f}°")
        if first_unstable_step is not None:
            print(f"  First unstable: step {first_unstable_step}, Last unstable: step {last_unstable_step}")
            if episode_info['unstable_duration'] is not None:
                print(f"  Unstable duration: {episode_info['unstable_duration']} steps")
        
        # 保存完整角度数据（对所有有足够数据的episodes，不要求达到max_cycles）
        # 只要angles数据足够长（>= 10个点），就保存
        if len(angles) >= 10:
            episode_info['angles'] = [float(a) for a in angles]
        
        # 过滤1：排除package接触地面的episodes（真正失败）
        if package_contact_arr and idx < len(package_contact_arr):
            if package_contact_arr[idx]:
                msg = "Package touched ground (real failure)"
                print(f"  ✗ {msg}")
                episode_info['status'] = 'filtered'
                episode_info['filter_reason'] = msg
                episode_details.append(episode_info)
                continue
        
        # 过滤2：排除角度数据异常的（全0）
        if np.all(angles_array == 0):
            msg = "Invalid data (all zeros)"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        # 过滤3：排除提前摔倒的
        if term_step < max_cycles:
            msg = f"Early termination ({term_step} < {max_cycles})"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        # 过滤4：排除全程稳定的（从未超过6.5°）
        if max_angle < 6.5:
            msg = f"Never unstable (max {max_angle:.2f}° < 6.5°)"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        # 查找失效和恢复点（复用已计算的first_unstable_step）
        failure_step = first_unstable_step
        recovery_step = None
        
        # 从失效点之后找到第一个降到5°以下的点
        if failure_step is not None:
            for j in range(failure_step, len(angles)):
                if abs(angles[j]) < 5.0:
                    recovery_step = j
                    break
        
        episode_info['failure_step'] = failure_step
        episode_info['recovery_step'] = recovery_step
        
        print(f"  Failure at step: {failure_step}, Recovery at step: {recovery_step}")
        
        # 只保留真正恢复的（既失稳过，又恢复了）
        if failure_step is not None and recovery_step is not None:
            recovery_time = recovery_step - failure_step
            episode_info['recovery_time'] = recovery_time
            episode_info['status'] = 'recovered'
            episode_info['filter_reason'] = f"RECOVERED! Recovery time: {recovery_time} steps"
            print(f"  ✓ {episode_info['filter_reason']}")
            
            recovered.append({
                'episode_idx': idx,
                'angles': angles,
                'max_angle': np.max(angles_array),
                'failure_step': failure_step,
                'recovery_step': recovery_step,
                'recovery_time': recovery_time,
                'first_unstable_step': first_unstable_step
            })
        else:
            if failure_step is None:
                msg = "Never failed (no angle > 6.5°)"
            else:
                msg = f"Failed but never recovered (no angle < 5.0° after step {failure_step})"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
        
        episode_details.append(episode_info)
    
    return recovered, episode_details


def plot_recovered_cases_only(recovered_cases, save_dir, config_info):
    """
    只绘制恢复的episodes，每5个episodes一张图
    """
    import os
    import matplotlib.pyplot as plt
    
    if not recovered_cases:
        print("⚠️  No recovered episodes to plot")
        return
    
    # 每5个episodes一张图
    episodes_per_plot = 5
    num_plots = (len(recovered_cases) + episodes_per_plot - 1) // episodes_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * episodes_per_plot
        end_idx = min(start_idx + episodes_per_plot, len(recovered_cases))
        cases_subset = recovered_cases[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(16, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(cases_subset)))
        
        for i, case in enumerate(cases_subset):
            ep_idx = case['episode_idx']
            angles = case['angles']
            x = np.arange(len(angles))
            y = np.abs(np.array(angles))  # 取绝对值
            
            # 绘制曲线
            ax.plot(x, y, label=f'Ep {ep_idx+1}', 
                   alpha=0.8, linewidth=2.5, color=colors[i])
            
            # 标注失效点（红×）
            if case['failure_step']:
                ax.scatter(case['failure_step'], abs(angles[case['failure_step']]), 
                          color='red', s=180, marker='x', zorder=10, linewidths=3)
            
            # 标注恢复点（绿●）
            if case['recovery_step']:
                ax.scatter(case['recovery_step'], abs(angles[case['recovery_step']]), 
                          color='green', s=180, marker='o', zorder=10, linewidths=2)
        
        # 阈值线
        ax.axhline(y=6.5, color='red', linestyle='--', linewidth=2.5, 
                   label='Failure Threshold (6.5°)', alpha=0.9)
        ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2.5, 
                   label='Recovery Threshold (5.0°)', alpha=0.9)
        ax.axvline(x=100, color='purple', linestyle=':', linewidth=2.5, 
                   label='Disturbance Start (Step 100)', alpha=0.9)
        
        ax.set_xlabel('Steps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Pole Angle (degrees, absolute)', fontsize=14, fontweight='bold')
        ax.set_title(
            f"Recovered Episodes (Set {plot_idx+1}/{num_plots}) - "
            f"Agent {config_info['target_agent']}, Magnitude {config_info['magnitude']}\n"
            f"Episodes {start_idx+1}-{end_idx} of {len(recovered_cases)} total recovered", 
            fontsize=16, fontweight='bold'
        )
        ax.legend(loc='best', fontsize=11, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        if num_plots > 1:
            save_path = os.path.join(save_dir, f"recovered_episodes_set{plot_idx+1}.png")
        else:
            save_path = os.path.join(save_dir, "recovered_episodes.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"✓ Plot {plot_idx+1}/{num_plots} saved: {save_path}")
    
    print(f"✓ Total recovered episodes: {len(recovered_cases)} in {num_plots} plot(s)")


def print_recovery_stats(recovered_cases):
    """打印恢复统计信息"""
    if not recovered_cases:
        print("\n⚠️  No recovered episodes found")
        return
    
    print(f"\n{'='*70}")
    print(f"RECOVERY STATISTICS")
    print(f"{'='*70}")
    print(f"Total recovered episodes: {len(recovered_cases)}")
    
    recovery_times = [c['recovery_time'] for c in recovered_cases if c['recovery_time']]
    if recovery_times:
        print(f"Average recovery time: {np.mean(recovery_times):.1f} steps")
        print(f"Recovery time range: {np.min(recovery_times)} - {np.max(recovery_times)} steps")
    
    max_angles = [c['max_angle'] for c in recovered_cases]
    print(f"Average max angle: {np.mean(max_angles):.2f}°")
    print(f"Max angle range: {np.min(max_angles):.2f}° - {np.max(max_angles):.2f}°")

    first_unstable_steps = [c['first_unstable_step'] for c in recovered_cases if c.get('first_unstable_step') is not None]
    if first_unstable_steps:
        print(f"Average first unstable step: {np.mean(first_unstable_steps):.1f}")
        print(f"First unstable step range: {np.min(first_unstable_steps)} - {np.max(first_unstable_steps)}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
