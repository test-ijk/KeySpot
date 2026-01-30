import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import List
from enum import Enum
import rich
import shutil
import os
import wandb
from datetime import datetime
from utils.notify import notify
import torch


class HydraStepType(Enum):
    train = "train"
    eval = "eval"
    bash = "bash"
    bark = "bark"
    parallel_gpu = "parallel_gpu"


@dataclass
class HydraStepConfig:
    type: HydraStepType
    multirun: bool
    args: List[str]
    config_name: str


@dataclass
class HydraCommandConfig:
    commands: List[str]


@dataclass
class WandbConfig:
    use_wandb: bool
    project: str


@dataclass
class HydraRunConfig:
    run_group: str
    save_group: str
    commands: List[str]
    steps: List[DictConfig]
    wandb: WandbConfig


@hydra.main(config_path="runs", config_name="default", version_base=None)
def main(config: HydraRunConfig):
    rich.print(config)
    if config.description == "_DEFAULT_DESCRIPTION_":
        raise ValueError(
            "不允许使用默认文件，务必复制并新建一个run config以带来最好可复现性。"
        )

    # 组合存储的文件夹
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(hydra_output_dir + "/train", exist_ok=True)
    os.makedirs(hydra_output_dir + "/eval", exist_ok=True)

    # 把latest改成当前时间
    current_time = datetime.now().strftime("%m%d/%H%M")
    if config.run_group == "latest":
        config.run_group = current_time
    if config.save_group == "latest":
        config.save_group = current_time

    # 组合命令
    def _from_step_to_command(step: HydraStepConfig) -> HydraCommandConfig:
        rich.print(step)
        if HydraStepType(step.type) == HydraStepType.train:
            file_cmd = "uv run src/train/train.py"
            config_cmd = f"--config-name={step.config_name}"
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)
            group_cmd = f"run_group={config.run_group} save_group={config.save_group}"
            return f"{file_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd}"
        elif HydraStepType(step.type) == HydraStepType.eval:
            file_cmd = "uv run src/eval/eval.py"
            config_cmd = f"--config-name={step.config_name}"
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)
            group_cmd = f"run_group={config.run_group} save_group={config.save_group}"
            return f"{file_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd}"
        elif HydraStepType(step.type) == HydraStepType.bash:
            return " ".join(step.args)
        elif HydraStepType(step.type) == HydraStepType.parallel_gpu:
            gpu_count = torch.cuda.device_count()
            para_cmds = []
            gpu_idx = 0
            for i, arg in enumerate(step.args):
                os.makedirs(hydra_output_dir + "/logs", exist_ok=True)
                para_cmds.append(
                    f"CUDA_VISIBLE_DEVICES={gpu_idx} nohup {arg} run_group={config.run_group} save_group={config.save_group} > {hydra_output_dir}/logs/{i}.out &"
                )
                gpu_idx += 1
                if gpu_idx >= gpu_count:
                    gpu_idx = 0
            return "\n".join(para_cmds)
        elif HydraStepType(step.type) == HydraStepType.bark:
            return f"bark||{step.args[0]}"
        return ""

    commands = []
    commands += [_from_step_to_command(step) for step in config.steps]
    rich.print(commands)

    # 保存命令供复现
    sh_reproduce = "\n".join(commands)
    with open(hydra_output_dir + "/reproduce.sh", "w") as f:
        f.write(sh_reproduce)

    # 复制train和eval的yaml配置文件
    for step in config.steps:
        if step.type == HydraStepType.train:
            shutil.copy(
                f"src/configs/train/{step.config_name}.yaml",
                hydra_output_dir + f"/train/{step.config_name}.yaml",
            )
        elif step.type == HydraStepType.eval:
            shutil.copy(
                f"src/configs/eval/{step.config_name}.yaml",
                hydra_output_dir + f"/eval/{step.config_name}.yaml",
            )

    # 调用wandb存储代码和配置文件
    if config.wandb.use_wandb:
        run_run = wandb.init(
            project=config.wandb.project,
            name=f"entrypoint_{config.run_group}",
            group=config.run_group,
            job_type="entrypoint",
            save_code=True,
            config=OmegaConf.to_container(config, resolve=True),
            notes=config.description,
        )
        run_run.log_code(hydra_output_dir)
        run_run.log_code(hydra_output_dir + "/reproduce.sh")
        run_run.log_code(hydra_output_dir + "/train")
        run_run.log_code(hydra_output_dir + "/eval")
        run_run.finish()

    # 执行代码
    for command in commands:
        rich.print(f"Running command: 【{command}】")
        if command.startswith("bark||"):
            notify(command.split("||")[1])
        else:
            os.system(command)


if __name__ == "__main__":
    main()
