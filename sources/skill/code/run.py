import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import rich
import os
import wandb
from datetime import datetime
from .utils.notify import notify


class HydraStepType(Enum):
    train = "train"
    eval = "eval"
    llm_eval = "llm_eval"
    bash = "bash"
    bark = "bark"
    render = "render"
    render_case = "render_case"


@dataclass
class HydraStepConfig:
    type: HydraStepType
    args: List[str]
    multirun: Optional[bool] = False
    config_name: Optional[str] = None


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
    steps: List[HydraStepConfig]
    wandb: WandbConfig

    use_custom_eval_configs: bool = False
    custom_eval_configs_folder: str = ""


@hydra.main(config_path="../0.run", config_name="default", version_base=None)
def main(config: HydraRunConfig):
    rich.print(config)


    current_time = datetime.now().strftime("%m%d/%H%M")
    if config.run_group == "latest":
        config.run_group = current_time
    if config.save_group == "latest":
        config.save_group = current_time

    def _from_step_to_command(step: HydraStepConfig) -> str:
        rich.print(step)
        group_cmd = (
            f"wandb.wandb_group={config.run_group} model.save_group={config.save_group}"
        )
        wandb_cmd = f"++wandb.wandb_project={config.wandb.project}"
        if HydraStepType(step.type) == HydraStepType.train:
            file_cmd = "uv run python -m sources.skill.code.train"
            config_cmd = (
                "--config-name='0.train'"
                if step.config_name is None
                else f"--config-name='{step.config_name}'"
            )
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)
            return f"{file_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd} {wandb_cmd}"
        elif HydraStepType(step.type) == HydraStepType.eval:
            file_cmd = "uv run python -m sources.skill.code.eval "
            multirun_cmd = "--multirun" if step.multirun else ""
            folder_cmd = (
                f"--config-path={config.custom_eval_configs_folder}"
                if config.use_custom_eval_configs
                else ""
            )
            args_cmd = " ".join(step.args)

            config_cmd = (
                "--config-name='default'"
                if step.config_name is None
                else f"--config-name='{step.config_name}'"
            )
            return f"{file_cmd} {folder_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd} {wandb_cmd}"
        elif HydraStepType(step.type) == HydraStepType.llm_eval:
            file_cmd = "uv run python -m sources.skill.code.llm_run "
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)

            return f"{file_cmd} {multirun_cmd} {args_cmd} {group_cmd} {wandb_cmd}"
        elif HydraStepType(step.type) == HydraStepType.render:
            file_cmd = "uv run python -m sources.skill.code.eval"
            config_cmd = "++eval_settings.functions.render=True"
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)

            return f"{file_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd} {wandb_cmd}"
        elif HydraStepType(step.type) == HydraStepType.render_case:
            file_cmd = "uv run src/case.py"
            config_cmd = "--config-name=rend"
            multirun_cmd = "--multirun" if step.multirun else ""
            args_cmd = " ".join(step.args)
            return f"{file_cmd} {config_cmd} {multirun_cmd} {args_cmd} {group_cmd} {wandb_cmd}"
        elif HydraStepType(step.type) == HydraStepType.bash:
            return " ".join(step.args)
        elif HydraStepType(step.type) == HydraStepType.bark:
            return f"bark||{step.args[0]}"
        return ""

    commands = []
    commands += [_from_step_to_command(step) for step in config.steps]
    print(commands)

    sh_reproduce = "\n".join(commands)
    current_date = datetime.now().strftime("%m-%d")
    current_time = datetime.now().strftime("%H-%M")
    hydra_output_dir = f"sources/skill/0.run/reproduce/{current_date}/{current_time}"
    os.makedirs(hydra_output_dir, exist_ok=True)

    with open(hydra_output_dir + "/reproduce.sh", "w") as f:
        f.write(sh_reproduce)
    #     with open(hydra_output_dir + "/default.yaml", "w") as f:
    #         f.write(OmegaConf.to_yaml(config))
    #     shutil.copy("src/run.py", hydra_output_dir + "/run.py")
    #     shutil.copy("src/train.py", hydra_output_dir + "/train.py")
    #     shutil.copy("src/eval.py", hydra_output_dir + "/eval.py")
    #     with open(hydra_output_dir + "/pure.sh", "w") as f:
    #         f.write(sh_reproduce)
    # except Exception as e:
    #     rich.print(e)

    if config.wandb.use_wandb:
        config_dict = OmegaConf.to_container(config, resolve=True)
        if type(config_dict) is not dict:
            raise ValueError("config_dict is not a dict")
        run_run = wandb.init(
            project=config.wandb.project,
            name=f"entrypoint_{config.run_group}",
            group=config.run_group,
            job_type="entrypoint",
            # save_code=True,
            config=config_dict,
        )
        run_run.finish()


    for command in commands:
        rich.print(f"Running command: 【{command}】")
        if command.startswith("bark||"):
            notify("done /" + command.split("||")[1])
        else:
            os.system(command)


if __name__ == "__main__":
    main()
