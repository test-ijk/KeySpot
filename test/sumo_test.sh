#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  A0 A1 A2 A3
  B0 B1 B2 B3
  C0 C1 C2 C3
  D0 D1 D2 D3
)

for t in "${TARGETS[@]}"; do
  [[ -z "${t// }" ]] && continue
  
  echo "===== Running target ${t} ====="
  echo "current target: ${t}"
  
  CMD=(
    uv run python -m sources.skill.code.eval
    --config-name=sumo
    algorithm=mappo
    "++environment.env_tweak.tweak_types=[]"
    ++environment.env_tweak.action_perturb_prob=0.2
    eval_scenario=sumo/raw
    wandb.wandb_group=250722_llm_env
    model.save_group=handpicked/sumo/max_reward
    ++wandb.wandb_project=mw_harl_new
    "++environment.env_tweak.perturb_targets=[${t}]"
  )
  
  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
done
