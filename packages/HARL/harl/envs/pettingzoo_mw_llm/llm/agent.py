import os
import sys
import pandas as pd
import numpy as np
from .advanced_slope_detection import advanced_slope_detection, process_lidar_frame
from typing import Optional
import json
# if len(sys.argv) > 1:
#     OBS_DIR = sys.argv[1]
# else:
#     OBS_PATH = os.path.join(os.path.dirname(__file__), "agent_0_observations.csv")
#     OBS_DIR = os.path.dirname(OBS_PATH)
# AGENT_FILES = [f for f in os.listdir(OBS_DIR) if f.endswith("_observations.csv")]

ENV_DESC1 = """
"""

ENV_DESC = """

"""

ENV_DESC_old = """

"""

ENV_DESC1 = """

"""

def get_agent_position(offset_x_list, idx):
    sorted_idx = np.argsort(offset_x_list)
    if idx == sorted_idx[0]:
        return "left"
    elif idx == sorted_idx[-1]:
        return "right"
    else:
        return "middle"


def semantic_observation(row, slope_info=None):

    lines = []

    angle = row["hull_angle"]
    # lines.append(

    px = row["package_x_offset"]
    lines.append(f"{-px:.2f}")
    # LIDAR
    lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
    lidar_avg = np.mean(lidar_vals)
    lidar_min = np.min(lidar_vals)
    lidar_max = np.max(lidar_vals)
    lidar_range = lidar_max - lidar_min

    return "\n".join(lines)


def initial_judgement(row, prev_row=None):

    angle = row["hull_angle"]
    angle_warn = ""
    if prev_row is not None:
        delta_angle = abs(angle - prev_row["hull_angle"])
        if delta_angle > 0.1:
            angle_warn = f"\n- head angle{delta_angle:.2f}。"

    if abs(angle) > 0.3:
        angle_warn += f"\n- head angle({angle:.2f})"

    px = row["package_x_offset"]
    px_warn = ""
    if prev_row is not None:
        delta_px = abs(px - prev_row["package_x_offset"])
        if delta_px > 0.05:
            px_warn = f"\n- x{delta_px:.2f}"

    if px > 0.5 or px < -0.5:
        px_warn += f"\n- leave({-px:.2f})"

    left_offset = row.get("left_neighbor_x_offset", 0)
    right_offset = row.get("right_neighbor_x_offset", 0)
    px_warn += "\n"
    if left_offset != 0:
        px_warn += f"- left{abs(left_offset):.2f}"
    if right_offset != 0:
        px_warn += f"- right{abs(right_offset):.2f}"
    neighbor_warn = ""

    if left_offset == 0:
        neighbor_warn += " "
    elif abs(left_offset) < 0.25:
        neighbor_warn += f"\n- left({left_offset:.2f})"
    if right_offset == 0:
        neighbor_warn += " "
    elif abs(right_offset) > 0.45:
        neighbor_warn += f"\n- right({right_offset:.2f})"
    return " ".join([angle_warn, px_warn, neighbor_warn]).strip()


def combine_judgement(lidar_vals, slope_info):
    lidar_min = min(lidar_vals)
    if lidar_min < 0.2 and slope_info["status"] == "flat":
        return f"LIDAR min{lidar_min:.2f}m"
    if slope_info["status"] == "flat" and slope_info.get("info", ""):
        return f"flat{slope_info['info']}"
    if slope_info["status"] == "flat":
        return "flat"

    stat = f"(delta_y{(slope_info.get('avg_delta_y') or 0):.4f},max{(slope_info.get('max_delta_y') or 0):.4f},min{(slope_info.get('min_delta_y') or 0):.4f})"
    return f"{slope_info['status']}，{slope_info['info']} {stat}"


def output_prompt(step, agent_rows, prev_agent_rows):
    prompt = [ENV_DESC, f"Step {step}："]
    for agent_id, row in agent_rows.items():

        slope_info = row.get("slope_info", None)
        prompt.append(f"\n【Agent {agent_id}】")
        prompt.append("：" + semantic_observation(row, slope_info))
        prompt.append(
            "" + initial_judgement(row, prev_agent_rows.get(agent_id))
        )
        prompt.append(
            ""
            + combine_judgement(
                [float(row[f"lidar_{j}"]) for j in range(10)], slope_info or {}
            )
        )
        prompt.append("30。\n")
    return "\n".join(prompt)


def generate_prompt_from_obs_dir(obs_dir: str) -> str:

    agent_files = [f for f in os.listdir(obs_dir) if f.endswith("_observations.csv")]
    agent_dfs = [pd.read_csv(os.path.join(obs_dir, f)) for f in agent_files]
    steps = agent_dfs[0].shape[0]
    agent_names = [f.split("_")[1] for f in agent_files]
    agent_lidar_history = {name: [] for name in agent_names}
    all_prompts = []
    for step in range(steps):
        for i, df in enumerate(agent_dfs):
            row = df.iloc[step]
            agent_name = agent_names[i]
            lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
            channel_data = process_lidar_frame(lidar_vals)
            delta_ys = [d["height_diff"] if d else 0 for d in channel_data]
            agent_lidar_history[agent_name].append(delta_ys)
        if step % 20 == 0:
            agent_rows = {}
            prev_agent_rows = {}
            for i, df in enumerate(agent_dfs):
                row = df.iloc[step].copy()
                agent_name = agent_names[i]
                frame_history = agent_lidar_history[agent_name][-5:]
                lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
                slope_info = advanced_slope_detection(
                    lidar_vals, frame_history=frame_history
                )
                row["slope_info"] = slope_info
                agent_rows[agent_name] = row
                if step > 0:
                    prev_agent_rows[agent_name] = agent_dfs[i].iloc[step - 1]
            all_prompts.append(output_prompt(step, agent_rows, prev_agent_rows))
    return "\n".join(all_prompts)


def generate_markdown_per_agent(obs_dir: str) -> None:

    import os

    agent_files = [f for f in os.listdir(obs_dir) if f.endswith("_observations.csv")]
    agent_dfs = [pd.read_csv(os.path.join(obs_dir, f)) for f in agent_files]
    steps = agent_dfs[0].shape[0]
    agent_names = [f.split("_")[1] for f in agent_files]
    agent_lidar_history = {name: [] for name in agent_names}
    result_dir = os.path.join(obs_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    for step in range(steps):
        for i, df in enumerate(agent_dfs):
            row = df.iloc[step]
            agent_name = agent_names[i]
            lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
            channel_data = process_lidar_frame(lidar_vals)
            delta_ys = [d["height_diff"] if d else 0 for d in channel_data]
            agent_lidar_history[agent_name].append(delta_ys)
        if step % 20 == 0:
            for i, df in enumerate(agent_dfs):
                row = df.iloc[step].copy()
                agent_name = agent_names[i]
                frame_history = agent_lidar_history[agent_name][-5:]
                lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
                slope_info = advanced_slope_detection(
                    lidar_vals, frame_history=frame_history
                )
                row["slope_info"] = slope_info
                prev_row = df.iloc[step - 1] if step > 0 else None

                md_lines = [
                    f"# Multiwalker Agent {agent_name} Step {step}",
                    "\n## ",
                    ENV_DESC,
                    f"\n## \n{semantic_observation(row, slope_info)}",
                    f"\n## \n{initial_judgement(row, prev_row)}",
                    f"\n## \n{combine_judgement([float(row[f'lidar_{j}']) for j in range(10)], slope_info or {})}",
                    "\n> 30。\n",
                ]
                md_content = "\n".join(md_lines)
                md_path = os.path.join(result_dir, f"agent_{agent_name}_step_{step}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)


def generate_prompt(
    obses: "list[np.ndarray]",
    lidar_obses: Optional["list[list[np.ndarray]]"] = None,
    ref_v: Optional["list[float]"] = None,
) -> str:

    header = [
        "hull_angle",
        "hull_angular_velocity",
        "hull_velocity_x",
        "hull_velocity_y",
        "left_hip_angle",
        "left_hip_speed",
        "left_knee_angle",
        "left_knee_speed",
        "left_foot_ground_contact",
        "right_hip_angle",
        "right_hip_speed",
        "right_knee_angle",
        "right_knee_speed",
        "right_foot_ground_contact",
        "target_v",
        "target_h",
        "lidar_0",
        "lidar_1",
        "lidar_2",
        "lidar_3",
        "lidar_4",
        "lidar_5",
        "lidar_6",
        "lidar_7",
        "lidar_8",
        "lidar_9",
        "left_neighbor_x_offset",
        "left_neighbor_y_offset",
        "right_neighbor_x_offset",
        "right_neighbor_y_offset",
        "package_x_offset",
        "package_y_offset",
        "package_angle",
    ]
    agent_rows = {}
    prev_agent_rows = {}  
    for agent_id, obs in enumerate(obses):
        obs = np.asarray(obs).reshape(-1)  
        row = {k: float(obs[i]) for i, k in enumerate(header)}

        if lidar_obses is not None and agent_id < len(lidar_obses):
            lidar_obs = np.asarray(lidar_obses[agent_id]).reshape(-1)

            for j in range(10):
                if j < len(lidar_obs):
                    row[f"lidar_{j}"] = float(lidar_obs[j])

        # lidar
        lidar_vals = [row[f"lidar_{j}"] for j in range(10)]
        slope_info = advanced_slope_detection(lidar_vals, frame_history=None)
        row["slope_info"] = slope_info
        agent_rows[agent_id] = row
        # prev_agent_rows[agent_id] = None 
    prompt = [
        # ENV_DESC,
    ]
    for agent_id, row in agent_rows.items():
        slope_info = row.get("slope_info", None)
        prompt.append(f"\n【Agent {agent_id}】")
        prompt.append("" + semantic_observation(row, slope_info))
        prompt.append("" + initial_judgement(row, None))
        prompt.append(
            ""
            + combine_judgement(
                [float(row[f"lidar_{j}"]) for j in range(10)], slope_info or {}
            )
        )
        prompt.append("30 frames。\n")
    if ref_v is not None:
        prompt.append(f"speed：{json.dumps(ref_v)}")
    print("prompt: ", "\n".join(prompt))
    prompt = [ENV_DESC, "\n".join(prompt)]
    return "\n".join(prompt)


def main():
    if len(sys.argv) > 1:
        obs_dir = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == "--markdown":
            generate_markdown_per_agent(obs_dir)
            print(f"markdown{os.path.join(obs_dir, 'result')}")
            return
    else:
        obs_path = os.path.join(os.path.dirname(__file__), "agent_0_observations.csv")
        obs_dir = os.path.dirname(obs_path)
    result = generate_prompt_from_obs_dir(obs_dir)
    print(result)


if __name__ == "__main__":
    main()
