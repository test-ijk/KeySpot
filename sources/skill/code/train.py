import rich.pretty
import wandb
import hydra
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime
from .types.task.train_type import TrainConfig
import atexit


def _to_dict(cfg1) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict nor list")
    return dict_result


def _to_harl_dict(
    cfg: TrainConfig,
):
    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.scenario.name

    algo_args = cfg.algorithm_parameters
    env_args = cfg.environment_parameters

    run_name = f"[{algorithm_name}]<{scenario_name}>"
    env_tweaks = cfg.environment.env_tweak
    if hasattr(env_tweaks, "tweak_types"):
        for key in env_tweaks.tweak_types:
            if not key.startswith("_"):
                run_name += f"<{key}={_to_dict(env_tweaks)[key]}>"

    now_time = datetime.now().strftime("%m%d/%H%M")

    run_group = cfg.wandb.wandb_group
    if run_group == "latest":
        run_group = now_time

    save_group = cfg.model.save_group
    if save_group == "latest":
        save_group = now_time

    algo_args.logger.log_dir = f"./results/models/{save_group}"

    algo_dict = _to_dict(algo_args)
    env_dict = _to_dict(env_args)



    if env_name == "pettingzoo_mw" or env_name == "pettingzoo_mw_llm":
        from .types.environment.type_multiwalker import multiwalker_customize_dict

        algo_dict, env_dict = multiwalker_customize_dict(cfg, algo_dict, env_dict)
    elif env_name == "sumo" or env_name == "sumo_llm":
        from .types.environment.type_sumo import sumo_customize_dict

        algo_dict, env_dict = sumo_customize_dict(cfg, algo_dict, env_dict, save_group)
    elif env_name == "mapdn":
        from harl.envs.mapdn.mapdn_types import mapdn_customize_dict

        algo_dict, env_dict = mapdn_customize_dict(cfg, algo_dict, env_dict, save_group)


    if cfg.environment.env_tweak is not None:
        env_tweak = _to_dict(cfg.environment.env_tweak)
        for key in env_tweak.keys():
            if not key.startswith("_") and key != "tweak_types":
                env_dict[key] = env_tweak[key]
    if cfg.environment_scenario is not None:
        env_dict.update(_to_dict(cfg.environment_scenario))


    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }
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


@hydra.main(config_path="../1.config/task/train", config_name="sumo", version_base=None)
def main(cfg: TrainConfig):
    rich.pretty.pprint(_to_dict(cfg), expand_all=True)


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


    print("ENV_DICT!!")
    rich.print(env_dict)
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    @atexit.register
    def _cleanup():
        runner.close()
        wandb.finish()


    wandb.tensorboard.patch(root_logdir=runner.log_dir)  # type: ignore
    wandb.init(
        project=cfg.wandb.wandb_project,
        config={"original": _to_dict(cfg), "algo": algo_dict, "env": env_dict},
        sync_tensorboard=True,
        # name=run_name + f"_{ts}",
        group=run_group,
        job_type="train",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
        ],
    )
    wandb.define_metric(
        "logs/eval_average_episode_rewards/eval_average_episode_rewards/eval_average_episode_rewards",
        summary="max",
    )
    wandb.define_metric(
        "logs/eval_average_steps/eval_average_steps/eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_steps/eval_average_steps/eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_episode_rewards",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_episode_rewards",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_steps",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_terminate_x",
        summary="last",
    )
    wandb.define_metric(
        "eval_terminate_x",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_v_deviation",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_v_deviation",
        summary="min",
    )

    runner.run()

    runner.close()
    wandb.finish()


if __name__ == "__main__":
    main()
