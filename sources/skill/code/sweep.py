#!/usr/bin/env python3
"""
使用joblib并发执行wandb sweep agent命令
"""

import os
import argparse
from joblib import Parallel, delayed
from time import sleep


def run_wandb_agent(agent_id: str, delay: int = 0):
    """执行单个wandb agent命令，可延迟启动"""
    if delay > 0:
        print(f"延迟 {delay}s 启动 agent: {agent_id}")
        sleep(delay)
    cmd = f"wandb agent {agent_id}"
    print(f"启动agent: {agent_id}")
    os.system(cmd)
    print(f"完成agent: {agent_id}")


# wandb agent yuzh2001-iscas/HARL_mw-src/pfewgbxu
def main():
    parser = argparse.ArgumentParser(description="并发执行wandb sweep agents")
    parser.add_argument(
        "--agent_id",
        type=str,
        default="yuzh2001-iscas/mw_harl_exp_reward_new/wbsh6o0d",
        # default="yuzh2001-iscas/mapdn_hasac/6l0ev626",  #  yuzh2001-iscas/mapdn_hasac/h7hsvfdn
        help="wandb agent ID",
    )

    args = parser.parse_args()
    n_jobs = 12

    print(f"启动 {n_jobs} 个并发agent执行: {args.agent_id}")

    # 使用joblib并发执行，每个job依次延迟1s启动
    Parallel(n_jobs=n_jobs)(
        delayed(run_wandb_agent)(args.agent_id, i) for i in range(n_jobs)
    )

    print("所有agent执行完成")


if __name__ == "__main__":
    main()
