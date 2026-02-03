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

    rich.print(f"Exporting gif for {config_name}")
    
    try:
        base_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        gif_dir = os.path.join(base_dir, "./videos/")
    except ValueError:

        gif_dir = "./results/renders/"
    
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):

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


        clip = VideoFileClip(gif_path)
        clip.write_videofile(
            os.path.join(
                gif_folder,
                f"{config_name}_[{rewards:.2f}]_{i}.mp4",
            ),
            codec="libx264",
            logger=None,
        )

        os.remove(gif_path)


def eval(
    config: EvalConfig,
):
    rich.print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    def _modify_algo_and_env_dict():
        algo_dict["train"]["model_dir"] = checkpoint_path 

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

        if (this_env_is_mw_series) and algo_dict["train"].get(
            "num_env_steps"
        ) is not None:
            algo_dict["train"]["num_env_steps"] = 1 

        if this_env_is_mw_series:
            env_dict["custom"]["is_eval"] = True
            env_dict["custom"]["eval_disturb"] = _to_dict(config.eval_scenario).get(
                "disturbances", []
            )

    _modify_algo_and_env_dict()

    # rich.pretty.pprint(algo_dict, expand_all=True)
    # rich.pretty.pprint(env_dict, expand_all=True)

    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)
    print("there!!!!!!!!!!!!!!!!!!!!!!!")


    try:
        print("\n--- Listing Environment Object Attributes ---")
    
        env_attributes = dir(runner.envs)
        rich.print(env_attributes)

        print("---------------------------\n")
                
      
    except Exception as e:
        print(f"Failed to print agents due to an error: {e}")
    
    # sys.stdout.flush()
    # sys.exit()

    @atexit.register
    def _cleanup():
        runner.close()
        # wandb.finish()

    is_online_policy = hasattr(runner, "logger")
    start_time = time.time()


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

        angle_arr = []
        package_contact_arr = []
        if is_online_policy:
            print("it is onpolicy")
            runner = cast(OnPolicyMARunner, runner)
            logger: PettingZooMWLogger = runner.logger
            logger.is_testing = (
                True 
            )
            runner.eval()
            assert runner.eval_envs is not None
            runner.eval_envs.reset()
            terminate_arr = logger.test_data.get("terminate_at", [])
            if this_env_is_mw_series:
                angle_arr = logger.test_data.get("angle_data", [])
                

                def convert_thread_data_to_episodes(thread_angle_data, terminate_arr, n_threads):
                    """
                    Args:
                        thread_angle_data: list of list, thread_angle_data[tid]
                        terminate_arr: list, terminate_arr[i] 
                        n_threads: int, 
                    
                    Returns:
                        episode_angles: list of list, episode_angles[i] 
                    """
                    total_episodes = len(terminate_arr)
                    episode_angles = []
                    
                    thread_cursors = [0] * n_threads  
                    
                    for ep_idx in range(total_episodes):
                        thread_id = ep_idx % n_threads  
                        start = thread_cursors[thread_id]
                        end = start + terminate_arr[ep_idx]

                        if thread_id < len(thread_angle_data) and end <= len(thread_angle_data[thread_id]):
                            ep_angles = thread_angle_data[thread_id][start:end]
                            episode_angles.append(ep_angles)
                            thread_cursors[thread_id] = end
                        else:
                            episode_angles.append([])
                    
                    return episode_angles
                
                n_threads = logger.algo_args["eval"]["n_eval_rollout_threads"]
                angle_arr = convert_thread_data_to_episodes(angle_arr, terminate_arr, n_threads)
                
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


        terminate_cnt = 0
        package_x = []
        early_terminate_arr = []
        for i in range(len(terminate_arr)):
            if (
                terminate_arr[i] + 2 < config.environment.env_tweak.max_cycles
            ):
                terminate_cnt += 1
                early_terminate_arr.append(terminate_arr[i])
            if this_env_is_mw_series and is_online_policy:
                package_x.append(
                    logger.test_data["package_x"][i]
                    if is_online_policy
                    else runner.episode_xs[i]  # type: ignore
                )

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

                    threshold = 6.5
                    count_above_threshold = sum(1 for angle in angle_flatten if angle > threshold)
                    percentage_above = (count_above_threshold / len(angle_flatten)) * 100

                    with open(filepath, 'w') as f:
                        f.write(f"report\n")
                        f.write(f"==============\n")
                        f.write(f"all: {len(angle_flatten)}\n")
                        f.write(f"threshold: {threshold}\n")
                        f.write(f"over: {count_above_threshold}\n")
                        f.write(f"rate: {percentage_above:.2f}%\n\n")
                        f.write("over details:\n")

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

    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, episode_angles in enumerate(angle_data):
        x = np.arange(len(episode_angles))
        y = np.array(episode_angles)
        color = color_cycle[i % len(color_cycle)]
        plt.plot(x, y, label=f'Episode {i+1}', color=color, alpha=0.7)
    
    plt.axhline(y=6.5, color='red', linestyle='--', linewidth=2, label='Failure Threshold (6.5°)')
    plt.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, label='Recovery Threshold (5.0°)')
    
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

        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # time.sleep(10000)

    print("start!!!!!!!!!!!!!!!!!!!!!!!!")
    result = eval(cfg)

    if result is not None:
        import json
        from datetime import datetime

        now = datetime.now()
        mmdd = now.strftime("%m%d")
        hhmm = now.strftime("%H%M%S")

        save_dir = f"./results/runs/{mmdd}/{hhmm}"
        os.makedirs(save_dir, exist_ok=True)


        result = convert_np(result)
        
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
        
        if not angles or len(angles) < 10:
            msg = f"Too few data points: {len(angles) if angles else 0}"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        angles_array = np.abs(np.array(angles))  
        max_angle = np.max(angles_array)
        std_angle = np.std(angles_array)
        episode_info['max_angle'] = float(max_angle)
        episode_info['std_angle'] = float(std_angle)
        
        first_unstable_step = None
        last_unstable_step = None
        for j, angle in enumerate(angles):
            if abs(angle) > 6.5:
                if first_unstable_step is None:
                    first_unstable_step = j
                last_unstable_step = j 
        
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
        

        if len(angles) >= 10:
            episode_info['angles'] = [float(a) for a in angles]
        
        if package_contact_arr and idx < len(package_contact_arr):
            if package_contact_arr[idx]:
                msg = "Package touched ground (real failure)"
                print(f"  ✗ {msg}")
                episode_info['status'] = 'filtered'
                episode_info['filter_reason'] = msg
                episode_details.append(episode_info)
                continue

        if np.all(angles_array == 0):
            msg = "Invalid data (all zeros)"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue

        if term_step < max_cycles:
            msg = f"Early termination ({term_step} < {max_cycles})"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        if max_angle < 6.5:
            msg = f"Never unstable (max {max_angle:.2f}° < 6.5°)"
            print(f"  ✗ {msg}")
            episode_info['status'] = 'filtered'
            episode_info['filter_reason'] = msg
            episode_details.append(episode_info)
            continue
        
        failure_step = first_unstable_step
        recovery_step = None
        
        if failure_step is not None:
            for j in range(failure_step, len(angles)):
                if abs(angles[j]) < 5.0:
                    recovery_step = j
                    break
        
        episode_info['failure_step'] = failure_step
        episode_info['recovery_step'] = recovery_step
        
        print(f"  Failure at step: {failure_step}, Recovery at step: {recovery_step}")
        
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

    import os
    import matplotlib.pyplot as plt
    
    if not recovered_cases:
        print("⚠️  No recovered episodes to plot")
        return
    
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
            y = np.abs(np.array(angles))  
            
            ax.plot(x, y, label=f'Ep {ep_idx+1}', 
                   alpha=0.8, linewidth=2.5, color=colors[i])

            if case['failure_step']:
                ax.scatter(case['failure_step'], abs(angles[case['failure_step']]), 
                          color='red', s=180, marker='x', zorder=10, linewidths=3)
            
            if case['recovery_step']:
                ax.scatter(case['recovery_step'], abs(angles[case['recovery_step']]), 
                          color='green', s=180, marker='o', zorder=10, linewidths=2)
        
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
        
        if num_plots > 1:
            save_path = os.path.join(save_dir, f"recovered_episodes_set{plot_idx+1}.png")
        else:
            save_path = os.path.join(save_dir, "recovered_episodes.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"✓ Plot {plot_idx+1}/{num_plots} saved: {save_path}")
    
    print(f"✓ Total recovered episodes: {len(recovered_cases)} in {num_plots} plot(s)")


def print_recovery_stats(recovered_cases):
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
