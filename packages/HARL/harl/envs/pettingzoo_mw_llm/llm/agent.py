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
ENV_DESC1 = """环境简介：Multiwalker 
环境，多个双足机器人协作搬运包裹，需跨越复杂地形（如坡道、障碍等），整体向右前进
任务简介：保持队形、稳定搬运包裹，避免掉队、拥挤和跌倒，顺利通过地形。
当前输出说明：本帧为每50帧采样一次的观测结果。
【你的任务】为三个小人分配策略，以速度参考值的形式。速度取值范围是[0, 0.6]
输出格式：{"target_vs": [0.4, 0.4, 0.4]}
不输出任何其他文本，只输出上述选择；不使用markdown格式。使用双引号而非单引号。

* 速度参考值：0.4是正常速度，0.6是加速，0.1是减速。

## 参考信息
1. 包裹是一根长条
2. 相对包裹偏移如果绝对值在0.5以内，说明还支撑着包裹；如果值为正，则在包裹左侧，反之在右侧。

## 策略机制
1. 前进=向右。agent0在最左侧、agent2在最右侧。
2. 核心思路是维持三个机器人的相对距离比较稳定。
2.1 首先关注自然就是是否出现拥挤和掉队的提示。
2.1.1 如果某个机器人掉队了，请右侧的机器人慢下来等等它（调整到0.1），它和它左侧的机器人一起加速赶一赶（调整到0.6）；
2.1.2 如果某两个机器人拥挤了，那右侧的机器人们都走快点（调整到0.6），左侧的机器人们都走慢点（调整到0.1）。
2.2 如果出现特殊地形，考虑地形会对机器人行进速度的影响。例如如果在上坡，上坡的机器人的参考速度不变的情况下、实际的横向位移速度会变慢，那此时其他不在上坡的机器人就应该走慢一点（0.2）、这个上坡中的机器人应该走快点（0.6）。
2.3 拥挤和掉队应该优先于地形考虑；就算某个机器人在上坡，如果右侧机器人和这个上坡机器人拥挤了，右侧机器人也应该走快一点。
3. 如果没有上述情况，都选择正常策略[0.4, 0.4, 0.4]
* 速度参考值：0.4是正常速度，0.6是加速，0.1是减速。

## 信息
"""

ENV_DESC = """
你需要为Multiwalker环境中的三个双足机器人分配速度参考值，以实现它们保持队形、稳定搬运包裹，避免掉队、拥挤和跌倒，顺利通过复杂地形的任务。整体环境是多个双足机器人协作搬运一根长条包裹，需跨越复杂地形（如坡道、障碍等）并整体向右前进。

### 参考信息
1. 包裹是一根长条。
2. 相对包裹偏移值是相对包裹中心点，如果绝对值在0.5以内，说明还支撑着包裹；
3. 机器人与杆子的绝对距离不应大于0.4。
4. Agent0在左侧，Agent1在中间，Agent2在右侧。

### 策略机制
1. 前进方向为向右，agent0在最左侧、agent2在最右侧。
2. 核心思路是维持三个机器人的相对距离比较稳定，同时保证机器人与杆子的绝对距离不大于0.47。
    - 首先关注是否出现拥挤和掉队的提示：
        - 如果某个机器人掉队了，它右侧的所有机器人减速到0.3，它和它左侧的机器人加速到0.5。
        - 如果某两个机器人拥挤了，右侧的所有机器人加速到0.5，左侧的所有机器人们都减速到0.3。
        - 如果某两个机器人之间的距离大于0.5：右侧所有机器人减速到0.0，直到该距离变小到0.35以内，左侧机器人视情况加速。
    - 若某个机器人与杆子的绝对距离大于0.47：
        - 如果机器人的x坐标大于杆子的x坐标，则偏移值为正、说明机器人在杆子右侧。
        - 如果该机器人在杆子左侧，它和它左侧的机器人应该加速到0.5。
        - 如果该机器人在杆子右侧，它和它右侧的机器人应该减速到0.3。
    - 若位于中间的机器人与杆子的相对距离的绝对值大于0.2（三个机器人时，是agent1）：
        - 如果agent1的x坐标大于杆子的x坐标，则偏移值为正、说明agent1在杆子右侧。
        - 如果该机器人在杆子左侧，右侧的机器人减速到0.3，它和它左侧的机器人加速到0.5。
        - 如果该机器人在杆子右侧，它和左侧的机器人减速到0.3，它右侧的机器人加速到0.5。
    - 如果出现特殊地形，考虑地形会对机器人行进速度的影响。
        - 如果某个机器人在上坡，它应该加速到0.5，它左侧不在上坡的机器人减速到0.3，它右侧不在上坡的机器人减速到0.3
    - 冲突处理：
        - 如果某个机器人正在上坡，则不应该降低这个机器人的速度
3. 如果没有上述情况，都选择正常策略[0.4, 0.4, 0.4]。
4. 如果可以，尽量不要让速度出现突变，控制在+-0.2比较好。

### 速度参考值说明
0.4是正常速度，0.6是加速，0.3是减速。


## 输出格式
请根据上述信息为三个机器人分配速度参考值，
输出为：{"target_vs": [0.4, 0.4, 0.4]}
使用json格式，不输出任何其他文本，只输出上述内容；
不使用markdown格式，使用双引号而非单引号。
务必输出为能直接被json.loads解析的json字符串。


## 信息
当前提供的是每50帧采样一次的观测结果：

"""

ENV_DESC_old = """
你需要为Multiwalker环境中的三个双足机器人分配速度参考值，以实现它们保持队形、稳定搬运包裹，避免掉队、拥挤和跌倒，顺利通过复杂地形的任务。整体环境是多个双足机器人协作搬运一根长条包裹，需跨越复杂地形（如坡道、障碍等）并整体向右前进。

### 参考信息
1. 包裹是一根长条。
2. 相对包裹偏移值是相对包裹中心点，如果绝对值在0.5以内，说明还支撑着包裹；
3. 机器人与杆子的绝对距离不应大于0.4。
4. Agent0在左侧，Agent1在中间，Agent2在右侧。

### 策略机制
1. 前进方向为向右，agent0在最左侧、agent2在最右侧。
2. 核心思路是维持三个机器人的相对距离比较稳定，同时保证机器人与杆子的绝对距离不大于0.47。
    - 首先关注是否出现拥挤和掉队的提示：
        - 如果某个机器人掉队了，它右侧的所有机器人减速到0.2，它和它左侧的机器人加速到0.6。
        - 如果某两个机器人拥挤了，右侧的所有机器人加速到0.6，左侧的所有机器人们都减速到0.2。
        - 如果某两个机器人之间的距离大于0.5：右侧所有机器人减速到0.0，直到该距离变小到0.35以内，左侧机器人视情况加速。
    - 若某个机器人与杆子的绝对距离大于0.47：
        - 如果机器人的x坐标大于杆子的x坐标，则偏移值为正、说明机器人在杆子右侧。
        - 如果该机器人在杆子左侧，它和它左侧的机器人应该加速到0.6
        - 如果该机器人在杆子右侧，它和它右侧的机器人应该减速到0.2
    - 若位于中间的机器人与杆子的相对距离的绝对值大于0.2（三个机器人时，是agent1）：
        - 如果agent1的x坐标大于杆子的x坐标，则偏移值为正、说明agent1在杆子右侧。
        - 如果该机器人在杆子左侧，右侧的机器人减速到0.2，它和它左侧的机器人加速到0.6。
        - 如果该机器人在杆子右侧，它和左侧的机器人减速到0.2，它右侧的机器人加速到0.6。
    - 如果出现特殊地形，考虑地形会对机器人行进速度的影响。
        - 如果某个机器人在上坡，它应该加速到0.7，它左侧不在上坡的机器人减速到0.3，它右侧不在上坡的机器人减速到0.1
    - 冲突处理：
        - 如果某个机器人正在上坡，则不应该降低这个机器人的速度
3. 如果没有上述情况，都选择正常策略[0.4, 0.4, 0.4]。
4. 如果可以，尽量不要让速度出现突变，控制在+-0.2比较好。

### 速度参考值说明
0.4是正常速度，0.7是加速，0.1是减速。


## 输出格式
请根据上述信息为三个机器人分配速度参考值，
输出为：{"target_vs": [0.4, 0.4, 0.4]}
使用json格式，不输出任何其他文本，只输出上述内容；
不使用markdown格式，使用双引号而非单引号。
务必输出为能直接被json.loads解析的json字符串。


## 信息
当前提供的是每50帧采样一次的观测结果：

"""

ENV_DESC1 = """
==========================  Speed-Controller Prompt  ==========================
你是一名机器人队列速度控制器，只输出 3 个介于 0~0.7 的十进制数字，
对应 Agent0、Agent1、Agent2 的 v_ref；数字之间用英文逗号分隔，禁止输出其它任何字符。

控制规则：
1. 设期望位置 p_des = [−0.35, 0.00, +0.50]  （单位：米）
2. 读取各机器人观测中 “相对杆子中心点的距离”，记为 p_real = [p0,p1,p2]
3. 计算误差 err_i = p_des_i − p_real_i
4. 令 v_base = 0.40
5. 令 k_p = 1.2
6. 若任一观测包含 “已离开包裹” 字样，则所有 v_ref=0.05
7. 否则 v_ref_i = v_base + k_p × err_i
8. 把 v_ref_i 裁剪到区间 [0.00, 0.70]，保留两位小数
9. 请根据上述信息为三个机器人分配速度参考值，输出为json格式！例如：{"target_vs": [0.4, 0.4, 0.4]}，不输出任何其他文本，只输出上述选择；不使用markdown格式，使用双引号而非单引号。不用```包裹。
10. 务必输出为能直接被json.loads解析的json字符串。

【PROMPT_INPUT】
"""
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
    """
    关键变量语义化描述，包括单位、参考系、统计量
    """
    lines = []
    # 头部角度
    angle = row["hull_angle"]
    # lines.append(
    #     f"头部角度: {angle:.2f} 弧度（0为水平，π/2为竖直，单位: 弧度，世界坐标系）"
    # )
    # x偏移
    px = row["package_x_offset"]
    lines.append(f"相对杆子中心点的距离: {-px:.2f}")
    # LIDAR
    lidar_vals = [float(row[f"lidar_{j}"]) for j in range(10)]
    lidar_avg = np.mean(lidar_vals)
    lidar_min = np.min(lidar_vals)
    lidar_max = np.max(lidar_vals)
    lidar_range = lidar_max - lidar_min
    # lines.append(
    #     f"LIDAR距离: 平均{lidar_avg:.2f}米，最小{lidar_min:.2f}米，最大{lidar_max:.2f}米，范围{lidar_range:.2f}米"
    # )
    # 统计坡道信息
    # if slope_info:
    #     lines.append(
    #         f"坡道检测统计: delta_y均值{(slope_info.get('avg_delta_y') or 0):.4f}，最大{(slope_info.get('max_delta_y') or 0):.4f}，最小{(slope_info.get('min_delta_y') or 0):.4f}，范围{(slope_info.get('height_range') or 0):.4f}，上坡通道数{(slope_info.get('total_uphill_channels') or 0)}"
    #     )
    #     lines.append(
    #         f"近距离LIDAR通道（<0.2米）: {slope_info.get('near_channels', [])}"
    #     )
    #     lines.append(f"远距离LIDAR通道（>0.8米）: {slope_info.get('far_channels', [])}")
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
        px_warn += f"\n- 已离开包裹({-px:.2f})，有脱离风险。"
    # 掉队/拥挤风险判断
    left_offset = row.get("left_neighbor_x_offset", 0)
    right_offset = row.get("right_neighbor_x_offset", 0)
    px_warn += "\n"
    if left_offset != 0:
        px_warn += f"- 左侧与邻居距离{abs(left_offset):.2f}"
    if right_offset != 0:
        px_warn += f"- 右侧与邻居距离{abs(right_offset):.2f}"
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
    lidar_obses: Optional["list[list[np.ndarray]]"] = None,
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
        # ENV_DESC,
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
    print("prompt: ", "\n".join(prompt))
    prompt = [ENV_DESC, "\n".join(prompt)]
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
