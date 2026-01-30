import torch as th
import os
import yaml
from tensorboardX import SummaryWriter

from mapdn.models.model_registry import Model, Strategy
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn.utilities.util import convert, dict2str
from mapdn.utilities.trainer import PGTrainer

import wandb
from datetime import datetime

import hydra
from omegaconf import OmegaConf
import rich

from train_hydra_type import TrainHydraEntryConfig, TrainConfig


@hydra.main(
    config_path="../configs/train", config_name="case33.yaml", version_base=None
)
def run(configs: TrainHydraEntryConfig):
    rich.print(OmegaConf.to_container(configs, resolve=True))
    # 1. 检查配置是否合法
    OmegaConf.structured(TrainConfig(**configs.train_config))

    # 2. 运行
    rich.print(OmegaConf.to_container(configs, resolve=True))
    argv = configs.train_config
    global_prefix = "./mapdn"

    # load env args
    with open(f"{global_prefix}/args/env_args/{argv.env}.yaml", "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    if configs.train_config.alias == "pv_obs":
        env_config_dict["state_space"] = [
            "pv",
            "demand",
            "reactive",
            "vm_pu",
            "va_degree",
            "pv_disturbance_active",
        ]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = argv.scenario
    env_config_dict["data_path"] = "/".join(data_path)
    net_topology = argv.scenario

    # set the action range
    assert net_topology in [
        "case33_3min_final",
        "case141_3min_final",
        "case322_3min_final",
    ], f"{net_topology} is not a valid scenario."
    if argv.scenario == "case33_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif argv.scenario == "case141_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif argv.scenario == "case322_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    assert argv.mode in [
        "distributed",
        "decentralised",
    ], "Please input the correct mode, e.g. distributed or decentralised."
    env_config_dict["mode"] = argv.mode
    env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

    # load default args
    with open(f"{global_prefix}/args/default.yaml", "r") as f:
        default_config_dict = yaml.safe_load(f)

    # load alg args
    with open(f"{global_prefix}/args/alg_args/{argv.alg}.yaml", "r") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]
        alg_config_dict["action_scale"] = env_config_dict["action_scale"]
        alg_config_dict["action_bias"] = env_config_dict["action_bias"]

    log_name = "-".join(
        [
            # argv.env,
            net_topology,
            # argv.mode,
            argv.alg,
            argv.voltage_barrier_type,
            argv.alias,
        ]
    )

    wandb_name = "-".join(
        [
            f"[{datetime.now().strftime('%m%d-%H%M')}]",
            # net_topology,
            # argv.mode,
            argv.alg,
            # argv.voltage_barrier_type,
            argv.alias,
        ]
    )

    alg_config_dict = {**default_config_dict, **alg_config_dict}

    if configs.alg_override is not None:
        alg_config_dict["max_steps"] = configs.alg_override.max_steps

    # define envs
    if configs.train_config.alias == "original":
        configs.disturbances = None
    env = VoltageControl(
        env_config_dict,
        configs.disturbances if configs.disturbances is not None else [],
    )

    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    args = convert(alg_config_dict)

    # ============wandb============
    wandb_run = wandb.init(
        project="mapdn_new",
        config={
            "hp": OmegaConf.to_container(configs, resolve=True),
            "train": alg_config_dict,
        },
        sync_tensorboard=True,
        name=wandb_name,
        save_code=True,
        group=configs.run_group,
        tags=[configs.train_config.alg, configs.train_config.alias],
        job_type="train",
    )

    # define the save path
    if argv.save_path[-1] == "/":
        save_path = argv.save_path
    else:
        save_path = argv.save_path + "/"

    # create the save folders
    if "model_save" not in os.listdir(save_path):
        os.mkdir(save_path + "model_save")
    if "tensorboard" not in os.listdir(save_path):
        os.mkdir(save_path + "tensorboard")
    if log_name not in os.listdir(save_path + "model_save/"):
        os.mkdir(save_path + "model_save/" + log_name)
    if log_name not in os.listdir(save_path + "tensorboard/"):
        os.mkdir(save_path + "tensorboard/" + log_name)
    else:
        path = save_path + "tensorboard/" + log_name
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # create the logger
    logger = SummaryWriter(save_path + "tensorboard/" + log_name)

    model = Model[argv.alg]

    strategy = Strategy[argv.alg]

    if strategy == "pg":
        train = PGTrainer(args, model, env, logger)
    elif strategy == "q":
        raise NotImplementedError("This needs to be implemented.")
    else:
        raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

    with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
        alg_args2str = dict2str(alg_config_dict, "alg_params")
        env_args2str = dict2str(env_config_dict, "env_params")
        file.write(alg_args2str + "\n")
        file.write(env_args2str + "\n")

    for i in range(args.train_episodes_num):
        stat = {}
        train.run(stat, i)
        train.logging(stat)
        if i % args.save_model_freq == args.save_model_freq - 1:
            train.print_info(stat)
            os.makedirs(
                save_path + f"models/{configs.save_group}/{log_name}", exist_ok=True
            )
            th.save(
                {"model_state_dict": train.behaviour_net.state_dict()},
                save_path + f"models/{configs.save_group}/" + log_name + "/model.pt",
            )
            print("The model is saved!\n")

    logger.close()

    # requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/训练完成")
    wandb_run.log_model(
        path=save_path + "models/" + log_name + "/model.pt", name=wandb_name
    )
    wandb_run.finish()


if __name__ == "__main__":
    run()
