#!/usr/bin/env python3

import os
import argparse
from joblib import Parallel, delayed
from time import sleep


def run_wandb_agent(agent_id: str, delay: int = 0):

    if delay > 0:
        print(f" {delay}s,: {agent_id}")
        sleep(delay)
    cmd = f"wandb agent {agent_id}"
    print(f"agent: {agent_id}")
    os.system(cmd)
    print(f"agent: {agent_id}")


def main():
    parser = argparse.ArgumentParser(description="wandb sweep agents")
    parser.add_argument(
        "--agent_id",
        type=str,
        default="",
        help="wandb agent ID",
    )

    args = parser.parse_args()
    n_jobs = 12

    print(f"{n_jobs} : {args.agent_id}")


    Parallel(n_jobs=n_jobs)(
        delayed(run_wandb_agent)(args.agent_id, i) for i in range(n_jobs)
    )

    print("done")


if __name__ == "__main__":
    main()
