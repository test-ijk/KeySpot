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

wandb_results = []
max_cycles = 10000


def export_gif(
    config_name, frames_arr, rewards_arr, globalConfig: hydra_type.EntrypointConfig
):

    rich.print(f"Exporting gif for {config_name}")
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos/"
    )
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):

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
 
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)
    results = []


    for scenario_name, scenario in config.disturbances.items():
        scenario: hydra_type.ScenarioConfig = scenario

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
    rich.print(
        Panel(
            f"Checkpoint Path: {checkpoint_path}\nScenario Name: {eval_scenario.name}",
            title="Evaluation Info",
        )
    )

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

        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("episode_length") is not None
        ):
            algo_args.train.episode_length = globalConfig.env_tweak.max_cycles
        env_args.max_cycles = globalConfig.env_tweak.max_cycles

        algo_args.train.model_dir = checkpoint_path

        algo_args.eval.n_eval_rollout_threads = globalConfig.basic_config.eval_threads
        algo_args.eval.eval_episodes = globalConfig.basic_config.eval_episodes

        # gpu
        algo_args.device.cuda = globalConfig.basic_config.use_gpu


        if globalConfig.basic_config.render:
            algo_args.render.use_render = True
            algo_args.render.render_episodes = globalConfig.basic_config.eval_episodes


        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1

        algo_dict = _to_dict(algo_args)
        del algo_dict["name"]

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]

        for key, value in _to_dict(globalConfig.env_tweak).items():
            if value is not None:
                env_dict[key] = value
        env_dict["custom"]["eval_disturb"] = _to_dict(eval_scenario)["disturbances"]
        env_dict["custom"]["is_eval"] = True
        del env_dict["tweak_types"]


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

            if rgb_array is not None and episode_obses_arr is not None:
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

            export_gif(
                config_name=config_name,
                frames_arr=rgb_array,
                rewards_arr=rewards_arr,
                globalConfig=globalConfig,
            )

            exit()
        else:
            has_logger = hasattr(runner, "logger")
            if has_logger:
                logger: PettingZooMWLogger = runner.logger
                logger.is_testing = (
                    True
                )
                runner.eval()
                terminate_arr = logger.test_data["terminate_at"]
                angle_arr = logger.test_data["angle_data"]
            else:
                logger = None
                runner.eval(1)
                terminate_arr = runner.eval_episode_lens
                angle_arr = runner.eval_episode_angles

            terminate_cnt = 0
            package_x = []
            for i in range(len(terminate_arr)):
                if (
                    terminate_arr[i] + 2 < globalConfig.env_tweak.max_cycles
                ):
                    terminate_cnt += 1
                package_x.append(
                    logger.test_data["package_x"][i]
                    if has_logger
                    else runner.episode_xs[i]
                )
            if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
                runner.eval_envs.close()
            runner.close()

            end_time = time.time()
            print(
                f"[{algorithm}]<{checkpoint_type}>_{eval_scenario.name} time: {end_time - start_time:.2f} seconds"
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
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f)
    return filepath


def load_eval_results(filepath: str) -> dict:
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
                fname = fname.replace("/", "_")
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
    json_dir = "./results/eval_results"
    os.makedirs(json_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H:%M")
    GlobalHydra.instance().clear()

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
            result_file_name = cfg.basic_config.result_file_name
            if result_file_name == "latest":
                latest_dir = os.path.join(json_dir, "latest")
                results = load_eval_results(
                    os.path.join(latest_dir, f"{algorithm}.json")
                )
            else:
                timestamp_dir = os.path.join(json_dir, result_file_name)
                results = load_eval_results(
                    os.path.join(timestamp_dir, f"{algorithm}.json")
                )
        else:
            results = run_evaluations(cfg, algorithm)

            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            save_eval_results(
                results,
                timestamp_dir,
                f"{algorithm}",
            )

            save_eval_results(results, latest_dir, f"{algorithm}")

        wandb_results = []
        rich.print(wandb_results)
        analyze_eval_results(results, cfg, algorithm)
        log_wandb(cfg)

    start_time = time.time()
    process_checkpoint(cfg.algorithm)
    end_time = time.time()
    print(f"{cfg.algorithm} time: {end_time - start_time:.2f} seconds")


    run.finish()
    # exit()


if __name__ == "__main__":
    main()
