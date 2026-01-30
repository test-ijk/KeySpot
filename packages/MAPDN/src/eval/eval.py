import torch
import yaml
from mapdn.models.model_registry import Model, Strategy
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn.utilities.util import convert
from mapdn.utilities.tester import PGTester
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import OmegaConf
import rich

from eval_hydra_type import EvalHydraEntryConfig, EvalConfig

import wandb
from datetime import datetime


@hydra.main(
    config_path="../configs/eval", config_name="case335.yaml", version_base=None
)
def run(configs: EvalHydraEntryConfig):
    # 1. 检查配置是否合法
    OmegaConf.structured(EvalConfig(**configs.eval_config))
    # for disturbance in configs.disturbances:
    # OmegaConf.structured(DisturbanceConfig(**disturbance))

    # 2. 读取参数
    rich.print(OmegaConf.to_container(configs, resolve=True))
    config_name = HydraConfig.get().job.config_name
    argv = configs.eval_config
    global_prefix = "./mapdn"

    log_name = "-".join(
        [
            # argv.env,
            argv.scenario,
            # argv.mode,
            argv.alg,
            argv.voltage_barrier_type,
            argv.alias,
        ]
    )
    date_stamp = datetime.now().strftime("%m%d-%H%M")
    wandb_name = "-".join(
        [
            f"[{date_stamp}]",
            # argv.scenario,
            # argv.mode,
            argv.alg,
            # argv.voltage_barrier_type,
            argv.alias,
        ]
    )

    print(f"Now testing: {wandb_name}")
    runa = wandb.init(
        project="mapdn_new",
        name=wandb_name,
        save_code=True,
        config=OmegaConf.to_container(configs, resolve=True),
        group=configs.run_group,
        tags=[configs.eval_config.alg, configs.eval_config.alias],
        job_type="eval",
    )
    wandb.define_metric("terminate_cnt", summary="min")

    # 3. ENV的参数
    def _read_env_args():
        with open(f"{global_prefix}/args/env_args/{argv.env}.yaml", "r") as f:
            env_config_dict = yaml.safe_load(f)["env_args"]
        data_path = env_config_dict["data_path"].split("/")
        data_path[-1] = argv.scenario
        env_config_dict["data_path"] = "/".join(data_path)

        # set the action range
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
        env_config_dict["episode_limit"] = 480  # for one-day test

        if configs.eval_config.alias == "pv_obs":
            env_config_dict["state_space"] = [
                "pv",
                "demand",
                "reactive",
                "vm_pu",
                "va_degree",
                "pv_disturbance_active",
            ]
        return env_config_dict

    env_config_dict = _read_env_args()

    # 5. 实例化ENV
    if not configs.disturbances:
        configs.disturbances = []

    env = VoltageControl(env_config_dict, configs.disturbances)

    # 4. 算法的参数
    def _read_general_args():
        with open(global_prefix + "/args/default.yaml", "r") as f:
            default_config_dict = yaml.safe_load(f)
        default_config_dict["max_steps"] = 480
        return default_config_dict

    default_config_dict = _read_general_args()

    def _read_alg_args():
        with open(global_prefix + "/args/alg_args/" + argv.alg + ".yaml", "r") as f:
            alg_config_dict = yaml.safe_load(f)["alg_args"]
        alg_config_dict["action_scale"] = env_config_dict["action_scale"]
        alg_config_dict["action_bias"] = env_config_dict["action_bias"]

        alg_config_dict = {**default_config_dict, **alg_config_dict}
        alg_config_dict["agent_num"] = env.get_num_of_agents()
        alg_config_dict["obs_size"] = env.get_obs_size()
        alg_config_dict["action_dim"] = env.get_total_actions()
        alg_config_dict["cuda"] = False  # gpu!

        return alg_config_dict

    alg_config_dict = _read_alg_args()

    args = convert(alg_config_dict)

    model = Model[argv.alg]

    if args.target:
        target_net = model(args)
        behaviour_net = model(args, target_net)
    else:
        behaviour_net = model(args)

    # 6. 读取模型checkpoint
    if argv.save_path[-1] == "/":
        save_path = argv.save_path
    else:
        save_path = argv.save_path + "/"

    LOAD_PATH = save_path + f"{configs.save_group}/" + log_name + "/model.pt"
    print(f"Loading model from {LOAD_PATH}")

    checkpoint = (
        torch.load(LOAD_PATH, map_location="cpu")
        if not args.cuda
        else torch.load(LOAD_PATH)
    )
    behaviour_net.load_state_dict(checkpoint["model_state_dict"])

    strategy = Strategy[argv.alg]
    assert strategy == "pg"
    test = PGTester(args, behaviour_net, env, argv.render)
    # env = VoltageControl(env_config_dict, configs.disturbances)
    # behaviour_net.load_state_dict(checkpoint["model_state_dict"])

    if argv.test_mode == "single":
        record = test.run(argv.test_day, 23, 2)
    elif argv.test_mode == "batch":
        record = test.batch_run(argv.eval_episodes)
        data = [v for v in record.values()]
        wandb.log(
            {
                "results_table": wandb.Table(
                    columns=["wandb_name", "config_name"] + list(record.keys()),
                    data=[[wandb_name, config_name] + data],
                )
            }
        )
        runa.finish()


if __name__ == "__main__":
    run()
