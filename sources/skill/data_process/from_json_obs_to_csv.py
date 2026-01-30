import json
import numpy as np
import csv
import os
import pandas as pd

position_prefix = "/root/proj/2507-multiwalker-harl/results/outputs/2025-07-17/11-03-25/data/[mappo]<pettingzoo_mw>_move<n_walkers=3><max_cycles=1000><reward_factor=0.5><terminate_reward=-10.0>"
json_position = f"{position_prefix}_episode_obses.json"
lidar_json_position = f"{position_prefix}_lidar_obs.json"


def get_observation_labels(obs_dim):
    """根据观测维度返回对应的标签"""
    if obs_dim == 24:
        # 基础版本 (multiwalker_custom.py)
        return (
            [
                "hull_angle",  # 0: 躯体角度
                "hull_angular_velocity",  # 1: 躯体角速度 (2.0 * angularVelocity / FPS)
                "hull_velocity_x",  # 2: 躯体x方向速度 (0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS)
                "hull_velocity_y",  # 3: 躯体y方向速度 (0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS)
                "left_hip_angle",  # 4: 左髋关节角度
                "left_hip_speed",  # 5: 左髋关节速度 (speed / SPEED_HIP)
                "left_knee_angle",  # 6: 左膝关节角度 (angle + 1.0)
                "left_knee_speed",  # 7: 左膝关节速度 (speed / SPEED_KNEE)
                "left_foot_ground_contact",  # 8: 左脚是否接触地面 (1.0 if contact else 0.0)
                "right_hip_angle",  # 9: 右髋关节角度
                "right_hip_speed",  # 10: 右髋关节速度 (speed / SPEED_HIP)
                "right_knee_angle",  # 11: 右膝关节角度 (angle + 1.0)
                "right_knee_speed",  # 12: 右膝关节速度 (speed / SPEED_KNEE)
                "right_foot_ground_contact",  # 13: 右脚是否接触地面 (1.0 if contact else 0.0)
            ]
            + [f"lidar_{i}" for i in range(10)]
        )  # 14-23: 10个激光雷达距离测量

    elif obs_dim == 26:
        # motor_stable版本 (添加了motor参数)
        return [
            "hull_angle",  # 0: 躯体角度
            "hull_angular_velocity",  # 1: 躯体角速度
            "hull_velocity_x",  # 2: 躯体x方向速度
            "hull_velocity_y",  # 3: 躯体y方向速度
            "left_hip_angle",  # 4: 左髋关节角度
            "left_hip_speed",  # 5: 左髋关节速度
            "left_knee_angle",  # 6: 左膝关节角度
            "left_knee_speed",  # 7: 左膝关节速度
            "left_foot_ground_contact",  # 8: 左脚是否接触地面
            "right_hip_angle",  # 9: 右髋关节角度
            "right_hip_speed",  # 10: 右髋关节速度
            "right_knee_angle",  # 11: 右膝关节角度
            "right_knee_speed",  # 12: 右膝关节速度
            "right_foot_ground_contact",  # 13: 右脚是否接触地面
            "speed_factor_hip",  # 14: 髋关节速度因子 (motor参数)
            "speed_factor_knee",  # 15: 膝关节速度因子 (motor参数)
        ] + [f"lidar_{i}" for i in range(10)]  # 16-25: 10个激光雷达距离测量

    elif obs_dim == 31:
        # 完整版本 (包括邻居和包裹信息)
        base_labels = get_observation_labels(26 if obs_dim > 26 else 24)
        neighbor_labels = [
            "left_neighbor_x_offset",  # 左邻居x偏移
            "left_neighbor_y_offset",  # 左邻居y偏移
            "right_neighbor_x_offset",  # 右邻居x偏移
            "right_neighbor_y_offset",  # 右邻居y偏移
            "package_x_offset",  # 包裹x偏移
            "package_y_offset",  # 包裹y偏移
            "package_angle",  # 包裹角度
        ]
        return base_labels + neighbor_labels

    elif obs_dim == 33:
        # 完整版本 (motor_stable + 邻居和包裹信息)
        return [
            "hull_angle",  # 0: 躯体角度
            "hull_angular_velocity",  # 1: 躯体角速度
            "hull_velocity_x",  # 2: 躯体x方向速度
            "hull_velocity_y",  # 3: 躯体y方向速度
            "left_hip_angle",  # 4: 左髋关节角度
            "left_hip_speed",  # 5: 左髋关节速度
            "left_knee_angle",  # 6: 左膝关节角度
            "left_knee_speed",  # 7: 左膝关节速度
            "left_foot_ground_contact",  # 8: 左脚是否接触地面
            "right_hip_angle",  # 9: 右髋关节角度
            "right_hip_speed",  # 10: 右髋关节速度
            "right_knee_angle",  # 11: 右膝关节角度
            "right_knee_speed",  # 12: 右膝关节速度
            "right_foot_ground_contact",  # 13: 右脚是否接触地面
            "speed_factor_hip",  # 14: 髋关节速度因子 (motor参数)
            "speed_factor_knee",  # 15: 膝关节速度因子 (motor参数)
            "lidar_0",  # 16: 激光雷达方向0 (正前方)
            "lidar_1",  # 17: 激光雷达方向1
            "lidar_2",  # 18: 激光雷达方向2
            "lidar_3",  # 19: 激光雷达方向3
            "lidar_4",  # 20: 激光雷达方向4
            "lidar_5",  # 21: 激光雷达方向5
            "lidar_6",  # 22: 激光雷达方向6
            "lidar_7",  # 23: 激光雷达方向7
            "lidar_8",  # 24: 激光雷达方向8
            "lidar_9",  # 25: 激光雷达方向9 (右侧)
            "left_neighbor_x_offset",  # 26: 左邻居x偏移
            "left_neighbor_y_offset",  # 27: 左邻居y偏移
            "right_neighbor_x_offset",  # 28: 右邻居x偏移
            "right_neighbor_y_offset",  # 29: 右邻居y偏移
            "package_x_offset",  # 30: 包裹x偏移
            "package_y_offset",  # 31: 包裹y偏移
            "package_angle",  # 32: 包裹角度
        ]

    else:
        # 通用标签
        return [f"obs_{i}" for i in range(obs_dim)]


def analyze_observation_at_75_percent(data):
    """分析75%位置的观测数据"""
    if not data or not data[0] or not data[0][0]:
        print("数据为空，无法分析")
        return

    # 获取第一个episode的数据进行分析
    episode = data[0]
    n_agents = len(episode)

    print("=== 75%位置观测数据分析 ===")

    for agent_idx in range(n_agents):
        agent_obs = episode[agent_idx]
        total_steps = len(agent_obs)

        # 计算75%位置
        step_75_percent = int(total_steps * 0.5)

        if step_75_percent < total_steps:
            obs_75_raw = agent_obs[step_75_percent]
            print(f"调试信息: Agent {agent_idx} 原始观测类型: {type(obs_75_raw)}")
            print(
                f"调试信息: 原始观测长度: {len(obs_75_raw) if hasattr(obs_75_raw, '__len__') else 0}"
            )

            # 检查是否需要展开嵌套结构
            if (
                isinstance(obs_75_raw, list)
                and len(obs_75_raw) == 1
                and isinstance(obs_75_raw[0], list)
            ):
                obs_75 = obs_75_raw[0]  # 展开嵌套
                print("调试信息: 检测到嵌套结构，已展开")
            else:
                obs_75 = obs_75_raw

            obs_dim = len(obs_75) if hasattr(obs_75, "__len__") else 0
            print(f"调试信息: 实际观测维度: {obs_dim}")
            if obs_dim > 0:
                print(
                    f"调试信息: 实际观测示例 (前5个): {obs_75[:5] if obs_dim >= 5 else obs_75}"
                )
            labels = get_observation_labels(obs_dim)

            print(
                f"\nAgent {agent_idx} - 75%位置观测 (第{step_75_percent}/{total_steps}步):"
            )
            print(f"观测维度: {obs_dim}")

            # 打印每个维度的值和含义
            for i, (label, value) in enumerate(zip(labels, obs_75)):
                # 确保label是字符串
                if not isinstance(label, str):
                    label = str(label)
                # 确保value是数值
                try:
                    value_str = f"{float(value):8.4f}"
                except (ValueError, TypeError):
                    value_str = f"{str(value):>8s}"
                print(f"  [{i:2d}] {label:25s}: {value_str}")

            # 分析一些关键指标
            print("\n  关键指标分析:")
            if obs_dim >= 14:
                print(
                    f"    躯体角度: {obs_75[0]:6.3f} rad ({np.degrees(obs_75[0]):6.1f}°)"
                )
                print(f"    x方向速度: {obs_75[2]:6.3f}")
                print(f"    y方向速度: {obs_75[3]:6.3f}")
                print(f"    左脚接触地面: {'是' if obs_75[8] > 0.5 else '否'}")
                print(f"    右脚接触地面: {'是' if obs_75[13] > 0.5 else '否'}")

                if obs_dim >= 26:  # motor_stable版本
                    print(f"    髋关节速度因子: {obs_75[14]:6.3f}")
                    print(f"    膝关节速度因子: {obs_75[15]:6.3f}")

                # 激光雷达数据分析
                lidar_start = 16 if obs_dim >= 26 else 14
                if obs_dim > lidar_start:
                    lidar_data = obs_75[lidar_start : lidar_start + 10]
                    print(f"    激光雷达平均距离: {np.mean(lidar_data):6.3f}")
                    print(f"    激光雷达最小距离: {np.min(lidar_data):6.3f}")


def compare_observations_across_time(data):
    """比较不同时间点的观测数据"""
    if not data or not data[0] or not data[0][0]:
        print("警告: 数据不完整，跳过时间对比分析")
        return

    episode = data[0]
    agent_0_obs = episode[0]
    total_steps = len(agent_0_obs)

    # 检查第一个观测的结构
    if total_steps > 0:
        first_obs_raw = agent_0_obs[0]
        print(f"\n调试信息: Agent 0 第一个观测的类型: {type(first_obs_raw)}")
        print(
            f"调试信息: 原始长度: {len(first_obs_raw) if hasattr(first_obs_raw, '__len__') else 'N/A'}"
        )

        # 检查是否需要展开嵌套结构
        if (
            isinstance(first_obs_raw, list)
            and len(first_obs_raw) == 1
            and isinstance(first_obs_raw[0], list)
        ):
            print("调试信息: 检测到嵌套结构，将在对比中展开")
        elif hasattr(first_obs_raw, "__len__") and len(first_obs_raw) > 0:
            print(
                f"调试信息: 第一个观测示例: {first_obs_raw[:5] if len(first_obs_raw) >= 5 else first_obs_raw}"
            )

    # 选择几个关键时间点进行比较
    time_points = [
        int(total_steps * 0.25),  # 25%
        int(total_steps * 0.50),  # 50%
        int(total_steps * 0.75),  # 75%
        total_steps - 1,  # 最后一步
    ]

    print("\n=== 不同时间点观测对比 (Agent 0) ===")

    key_indices = [0, 2, 3, 8, 13]  # 躯体角度, x速度, y速度, 左脚接触, 右脚接触
    key_labels = ["躯体角度", "x速度", "y速度", "左脚接触", "右脚接触"]

    print(f"{'时间点':>10s}", end="")
    for label in key_labels:
        print(f"{label:>12s}", end="")
    print()

    for i, step in enumerate(time_points):
        if step < total_steps:
            obs_raw = agent_0_obs[step]

            # 展开嵌套结构
            if (
                isinstance(obs_raw, list)
                and len(obs_raw) == 1
                and isinstance(obs_raw[0], list)
            ):
                obs = obs_raw[0]
            else:
                obs = obs_raw

            print(f"{step:>4d}({i * 25 + 25:2d}%)", end="")
            for idx in key_indices:
                if idx < len(obs):
                    try:
                        value = obs[idx]
                        if idx in [8, 13]:  # 接触标志
                            result = "是" if float(value) > 0.5 else "否"
                            print(f"{result:>12s}", end="")
                        else:
                            print(f"{float(value):>12.4f}", end="")
                    except (ValueError, TypeError, IndexError):
                        # 如果无法转换为数值，显示原始值
                        print(f"{str(obs[idx]):>12s}", end="")
                else:
                    print(f"{'N/A':>12s}", end="")
            print()


def extract_agents_lidar_to_csv(
    data, agent_indices=[1, 2, 3], output_dir="results/obs_csv"
):
    """提取指定agents的lidar数据并保存为CSV文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    if not data or len(data) == 0:
        print("错误：数据为空")
        return

    for agent_idx in agent_indices:
        if len(data[0]) <= agent_idx:
            print(f"错误：数据中没有agent {agent_idx}")
            continue

        # 获取第一个episode中指定agent的数据
        agent_obs = data[0][agent_idx]

        # 确定lidar数据的起始索引
        first_obs = agent_obs[0]
        if (
            isinstance(first_obs, list)
            and len(first_obs) == 1
            and isinstance(first_obs[0], list)
        ):
            first_obs = first_obs[0]

        obs_dim = len(first_obs)
        lidar_start = 16 if obs_dim >= 26 else 14  # 根据观测维度确定lidar起始位置

        # 准备CSV数据
        output_file = os.path.join(output_dir, f"agent_{agent_idx}_lidar_data.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            # 写入表头
            header = ["step"] + [f"lidar_{i}" for i in range(10)]
            writer.writerow(header)

            # 写入数据
            for step, obs in enumerate(agent_obs):
                if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], list):
                    obs = obs[0]

                lidar_data = obs[lidar_start : lidar_start + 10]
                row = [step] + lidar_data
                writer.writerow(row)

        print(f"已成功将agent {agent_idx}的lidar数据保存到 {output_file}")


def extract_all_observations_to_csv(data, output_dir="results/obs_csv"):
    """提取所有agents的完整观测数据并保存为CSV文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    if not data or len(data) == 0:
        print("错误：数据为空")
        return

    # 获取第一个episode中所有agents的数据
    episode = data[0]
    n_agents = len(episode)

    # 获取观测维度标签
    first_obs = episode[0][0]
    if (
        isinstance(first_obs, list)
        and len(first_obs) == 1
        and isinstance(first_obs[0], list)
    ):
        first_obs = first_obs[0]
    obs_dim = len(first_obs)
    labels = get_observation_labels(obs_dim)

    for agent_idx in range(n_agents):
        agent_obs = episode[agent_idx]

        # 准备CSV数据
        output_file = os.path.join(output_dir, f"agent_{agent_idx}_observations.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            # 写入表头
            header = ["step"] + labels
            writer.writerow(header)

            # 写入数据
            for step, obs in enumerate(agent_obs):
                if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], list):
                    obs = obs[0]
                row = [step] + obs
                writer.writerow(row)

        print(f"已成功将agent {agent_idx}的完整观测数据保存到 {output_file}")


# 读取JSON文件
print(f"正在读取文件: {json_position}")
try:
    with open(json_position, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("文件读取成功")
except FileNotFoundError:
    print(f"错误: 文件不存在 - {json_position}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"错误: JSON解析失败 - {e}")
    exit(1)

print("=== 数据结构维度分析 ===")
print(f"总体结构类型: {type(data)}")

# 验证数据结构
if not isinstance(data, list):
    print(f"错误: 数据应该是列表类型，但实际是: {type(data)}")
    exit(1)

if len(data) == 0:
    print("错误: 数据为空")
    exit(1)

print(f"第1层 - Episodes 数量: {len(data)}")

if len(data) > 0:
    first_episode = data[0]
    print(f"第2层 - 每个Episode中的Agents数量: {len(first_episode)}")

    if len(first_episode) > 0:
        first_agent = first_episode[0]
        print(f"第3层 - 每个Agent的观测步数: {len(first_agent)}")

        if len(first_agent) > 0:
            first_obs_raw = first_agent[0]
            print(f"第4层 - 原始观测数据类型: {type(first_obs_raw)}")
            print(
                f"第4层 - 原始观测长度: {len(first_obs_raw) if hasattr(first_obs_raw, '__len__') else 0}"
            )

            # 展开嵌套结构
            if (
                isinstance(first_obs_raw, list)
                and len(first_obs_raw) == 1
                and isinstance(first_obs_raw[0], list)
            ):
                first_obs = first_obs_raw[0]
                print("检测到嵌套结构，已展开")
            else:
                first_obs = first_obs_raw

            obs_dim = len(first_obs) if hasattr(first_obs, "__len__") else 0
            print(f"第4层 - 实际观测维度: {obs_dim}")

            # 显示观测维度标签
            labels = get_observation_labels(obs_dim)
            print(f"\n=== 观测维度说明 (共{obs_dim}维) ===")
            if obs_dim > 0:
                print(
                    f"观测数据示例 (前10个值): {first_obs[:10] if obs_dim >= 10 else first_obs}"
                )

                # 确保labels是正确的格式
                if not isinstance(labels, list):
                    print(f"错误: labels不是列表类型，而是: {type(labels)}")
                    labels = [f"obs_{i}" for i in range(obs_dim)]

                if len(labels) != obs_dim:
                    print(f"警告: 标签数量({len(labels)})与观测维度({obs_dim})不匹配")
                    labels = [f"obs_{i}" for i in range(obs_dim)]

                for i, (label, value) in enumerate(zip(labels, first_obs)):
                    # 确保label是字符串
                    if not isinstance(label, str):
                        label = str(label)
                    # 确保value是数值
                    try:
                        value_str = f"{float(value):8.4f}"
                    except (ValueError, TypeError):
                        value_str = f"{str(value):>8s}"
                    print(f"  [{i:2d}] {label:25s}: {value_str}")

print("\n=== 详细维度统计 ===")
for episode_idx, episode in enumerate(data):
    print(f"Episode {episode_idx}: {len(episode)} agents")
    for agent_idx, agent_obs in enumerate(episode):
        print(f"  Agent {agent_idx}: {len(agent_obs)} observations")
        if len(agent_obs) > 0:
            obs_dim = len(agent_obs[0])
            print(f"    每个观测维度: {obs_dim}")
    if episode_idx >= 2:  # 只展示前3个episode的详细信息
        print("  ... (更多episodes)")
        break

print("\n=== 总结 ===")
print("数据结构: [Episodes][Agents][TimeSteps][ObsDimension]")
print(
    f"具体维度: [{len(data)}][{len(data[0]) if data else 0}][变长][{len(data[0][0][0]) if data and data[0] and data[0][0] else 0}]"
)

# 提取所有agents的完整观测数据
extract_all_observations_to_csv(data)

# 进行75%位置的观测分析
analyze_observation_at_75_percent(data)

# 进行时间序列对比
compare_observations_across_time(data)

# 提取agent 0,1,2的lidar数据
extract_agents_lidar_to_csv(data, agent_indices=[0, 1, 2], output_dir="results/obs_csv")

# 读取JSON文件
with open(lidar_json_position, "r", encoding="utf-8") as f:
    lidar_data = json.load(f)

# 假设每个step的观测就是10个float
for agent_idx in range(3):
    rows = []
    for step_idx, step_obs in enumerate(lidar_data[0][agent_idx]):
        print(len(step_obs))
        # step_obs 应该是长度为10的list
        row = {"episode": 0, "step": step_idx}
        for i in range(10):
            row[f"lidar_{i}"] = step_obs[agent_idx][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = lidar_json_position.replace(".json", f"_agent{agent_idx}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Agent {agent_idx} 的CSV文件已保存到: {csv_path}")
    print(df.head())
