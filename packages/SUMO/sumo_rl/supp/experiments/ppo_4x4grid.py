import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl


if __name__ == "__main__":
    # Use:
    # ray[rllib]==2.7.0
    # numpy == 1.23.4
    # Pillow>=9.4.0
    # ray[rllib]==2.7.0
    # SuperSuit>=3.9.0
    # torch>=1.13.1
    # tensorflow-probability>=0.19.0
    ray.init()

    env_name = "4x4grid"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file="packages/SUMO/sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
                route_file="packages/SUMO/sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
                out_csv_name="results/sumo/4x4grid/ppo",
                use_gui=False,
                num_seconds=3600,
                sumo_warnings=False,
            )
        ),
    )

    from ray.air.integrations.wandb import WandbLoggerCallback

    wandb_logger = WandbLoggerCallback(
        project="sumo_harl_resco_4x4",
        name="ppo_4x4grid",
        group="ppo",
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="/root/proj/2507-multiwalker-harl/results/ray_results1/" + env_name,
        config=config.to_dict(),
        callbacks=[wandb_logger],
    )
