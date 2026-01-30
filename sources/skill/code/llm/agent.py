import os
import sys
import pandas as pd
import numpy as np
from .advanced_slope_detection import advanced_slope_detection, process_lidar_frame
from typing import Optional
import json
# 支持命令行参数传入观测数据目录
# if len(sys.argv) > 1:
#     OBS_DIR = sys.argv[1]
# else:
#     OBS_PATH = os.path.join(os.path.dirname(__file__), "agent_0_observations.csv")
#     OBS_DIR = os.path.dirname(OBS_PATH)
# AGENT_FILES = [f for f in os.listdir(OBS_DIR) if f.endswith("_observations.csv")]

# 环境与任务简介
ENV_DESC = """"""
# 计算agent在队伍中的相对位置
# offset_x越大越靠右，越小越靠左


def get_agent_position(offset_x_list, idx):
    sorted_idx = np.argsort(offset_x_list)
    if idx == sorted_idx[0]:
        return "最左（可能掉队）"
    elif idx == sorted_idx[-1]:
        return "最右（可能拥挤）"
    else:
        return "中间"


def semantic_observation(row, slope_info=None):

    lines = []
    # 头部角度
    angle = row["hull_angle"]
    lines.append(
        f"头部角度: {angle:.2f} 弧度（0为水平，π/2为竖直，单位: 弧度，世界坐标系）"
    )
    # x偏移
    px = row["package_x_offset"]
    lines.append(
        f"相对包裹X偏移: {px:.2f}（0为最左，1为最右，单位: 归一化，包裹参考系）"
    )
    # LIDAR
    lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
    lidar_avg = np.mean(lidar_vals)
    lidar_min = np.min(lidar_vals)
    lidar_max = np.max(lidar_vals)
    lidar_range = lidar_max - lidar_min
    lines.append(
        f"LIDAR距离: 平均{lidar_avg:.2f}米，最小{lidar_min:.2f}米，最大{lidar_max:.2f}米，范围{lidar_range:.2f}米"
    )
    # 统计坡道信息
    if slope_info:
        lines.append(
            f"坡道检测统计: delta_y均值{(slope_info.get('avg_delta_y') or 0):.4f}，最大{(slope_info.get('max_delta_y') or 0):.4f}，最小{(slope_info.get('min_delta_y') or 0):.4f}，范围{(slope_info.get('height_range') or 0):.4f}，上坡通道数{(slope_info.get('total_uphill_channels') or 0)}"
        )
        lines.append(
            f"近距离LIDAR通道（<0.2米）: {slope_info.get('near_channels', [])}"
        )
        lines.append(f"远距离LIDAR通道（>0.8米）: {slope_info.get('far_channels', [])}")
    return "\n".join(lines)


def initial_judgement(row, prev_row=None):
    # 头部角度突变
    angle = row["hull_angle"]
    angle_warn = ""
    if prev_row is not None:
        delta_angle = abs(angle - prev_row["hull_angle"])
        if delta_angle > 0.1:
            angle_warn = f"\n- 头部角度突变{delta_angle:.2f}，需注意。"
    # 头部角度绝对值
    if abs(angle) > 0.3:
        angle_warn += f"\n- 头部倾斜过大({angle:.2f})，注意平衡。"
    # x偏移突变
    px = row["package_x_offset"]
    px_warn = ""
    if prev_row is not None:
        delta_px = abs(px - prev_row["package_x_offset"])
        if delta_px > 0.05:
            px_warn = f"\n- x偏移突变{delta_px:.2f}，需注意。"
    # package_x_offset 只判断是否离开包裹
    if px > 0.5 or px < -0.5:
        px_warn += f"\n- 已离开包裹({px:.2f})，有脱离风险。"
    # 掉队/拥挤风险判断
    left_offset = row.get("left_neighbor_x_offset", 0)
    right_offset = row.get("right_neighbor_x_offset", 0)
    neighbor_warn = ""
    # 掉队风险（左侧距离大，且不是最左边）
    if left_offset == 0:
        neighbor_warn += " "
    elif abs(left_offset) < 0.25:
        neighbor_warn += f"\n- 左侧与邻居距离小({left_offset:.2f})，有拥挤风险。"
    # 拥挤风险（右侧距离大，且不是最右边）
    if right_offset == 0:
        neighbor_warn += " "
    elif abs(right_offset) > 0.45:
        neighbor_warn += f"\n- 右侧与邻居距离大({right_offset:.2f})，有掉队风险。"
    return " ".join([angle_warn, px_warn, neighbor_warn]).strip()


def combine_judgement(lidar_vals, slope_info):
    lidar_min = min(lidar_vals)
    # 优先障碍判断
    if lidar_min < 0.2 and slope_info["status"] == "平地":
        return f"前方有障碍但未检测到坡，LIDAR最小值{lidar_min:.2f}米"
    if slope_info["status"] == "平地" and slope_info.get("info", ""):
        return f"坡道检测：平地，{slope_info['info']}"
    if slope_info["status"] == "平地":
        return "坡道检测：平地"
    # 增加统计量输出
    stat = f"(delta_y均值{(slope_info.get('avg_delta_y') or 0):.4f}，最大{(slope_info.get('max_delta_y') or 0):.4f}，最小{(slope_info.get('min_delta_y') or 0):.4f})"
    return f"坡道检测：{slope_info['status']}，{slope_info['info']} {stat}"


def output_prompt(step, agent_rows, prev_agent_rows):
    prompt = [ENV_DESC, f"Step {step}："]
    for agent_id, row in agent_rows.items():
        # 取最近一次坡道检测统计
        slope_info = row.get("slope_info", None)
        prompt.append(f"\n【Agent {agent_id}】")
        prompt.append("观察语义化：" + semantic_observation(row, slope_info))
        prompt.append(
            "初步判断：" + initial_judgement(row, prev_agent_rows.get(agent_id))
        )
        prompt.append(
            "组合判断："
            + combine_judgement(
                [float(row[f"lidar_{j}"]) for j in range(10)], slope_info or {}
            )
        )
        prompt.append("当前输出说明：本帧为每30帧采样一次的观测结果。\n")
    return "\n".join(prompt)


def generate_prompt_from_obs_dir(obs_dir: str) -> str:
    """
    读取指定目录下所有agent的观测csv，生成多步prompt文本。
    Args:
        obs_dir (str): 观测csv文件所在目录
    Returns:
        str: 多步prompt文本（每步之间用\n分隔）
    """
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
    """
    读取指定目录下所有agent的观测csv，每20步为每个agent单独生成markdown文件，存储到obs_dir/result/agent_{id}_step_{step}.md。
    """
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
                # 生成markdown内容
                md_lines = [
                    f"# Multiwalker Agent {agent_name} Step {step}",
                    "\n## 环境与任务简介",
                    ENV_DESC,
                    f"\n## 观察语义化\n{semantic_observation(row, slope_info)}",
                    f"\n## 初步判断\n{initial_judgement(row, prev_row)}",
                    f"\n## 组合判断\n{combine_judgement([float(row[f'lidar_{j}']) for j in range(10)], slope_info or {})}",
                    "\n> 当前输出说明：本帧为每30帧采样一次的观测结果。\n",
                ]
                md_content = "\n".join(md_lines)
                md_path = os.path.join(result_dir, f"agent_{agent_name}_step_{step}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)


def generate_prompt(
    obses: "list[np.ndarray]",
    lidar_obses: Optional["list[np.ndarray]"] = None,
    ref_v: Optional["list[float]"] = None,
) -> str:
    """
    接收每个agent一帧的观测（顺序与csv一致），返回prompt字符串。
    Args:
        obses (list[np.ndarray]): 每个agent的观测，顺序与csv一致。
        lidar_obses (list[np.ndarray], optional): 每个agent的lidar观测，如果提供则替换obses中的lidar读数。
    Returns:
        str: prompt字符串
    """
    # csv表头顺序
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
    prev_agent_rows = {}  # 这里没有历史帧，先不支持
    for agent_id, obs in enumerate(obses):
        obs = np.asarray(obs).reshape(-1)  # 保证是一维
        row = {k: float(obs[i]) for i, k in enumerate(header)}

        # 如果提供了lidar_obses，则替换lidar读数
        if lidar_obses is not None and agent_id < len(lidar_obses):
            lidar_obs = np.asarray(lidar_obses[agent_id]).reshape(-1)
            # 替换lidar_0到lidar_9的值
            for j in range(10):
                if j < len(lidar_obs):
                    row[f"lidar_{j}"] = float(lidar_obs[j])

        # lidar
        lidar_vals = [row[f"lidar_{j}"] for j in range(10)]
        slope_info = advanced_slope_detection(lidar_vals, frame_history=None)
        row["slope_info"] = slope_info
        agent_rows[agent_id] = row
        # prev_agent_rows[agent_id] = None  # 没有历史
    prompt = [
        ENV_DESC,
    ]
    for agent_id, row in agent_rows.items():
        slope_info = row.get("slope_info", None)
        prompt.append(f"\n【Agent {agent_id}】")
        prompt.append("观察语义化：" + semantic_observation(row, slope_info))
        prompt.append("初步判断：" + initial_judgement(row, None))
        prompt.append(
            "组合判断："
            + combine_judgement(
                [float(row[f"lidar_{j}"]) for j in range(10)], slope_info or {}
            )
        )
        prompt.append("当前输出说明：本帧为每30帧采样一次的观测结果。\n")
    if ref_v is not None:
        prompt.append(f"当前的目标速度：{json.dumps(ref_v)}")
    return "\n".join(prompt)


def main():
    if len(sys.argv) > 1:
        obs_dir = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == "--markdown":
            generate_markdown_per_agent(obs_dir)
            print(f"已生成markdown到{os.path.join(obs_dir, 'result')}")
            return
    else:
        obs_path = os.path.join(os.path.dirname(__file__), "agent_0_observations.csv")
        obs_dir = os.path.dirname(obs_path)
    result = generate_prompt_from_obs_dir(obs_dir)
    print(result)


if __name__ == "__main__":
    main()
