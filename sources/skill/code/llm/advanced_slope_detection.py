import math
import numpy as np


# 数据解析 - 使用完整数据集的关键部分
def calculate_theta(i):
    return math.pi / 2 - 0.15 * i


def process_lidar_frame(lidar_readings):
    """处理单帧雷达数据，返回每个通道的距离和高度差"""
    h = lidar_readings[0]
    channel_data = []

    for i in range(10):
        d_i = lidar_readings[i]
        if d_i >= 1.0:  # 无回波，包括1.00的读数
            channel_data.append(None)
        else:
            theta = calculate_theta(i)
            delta_y = h - d_i * math.sin(theta)
            x = d_i * math.cos(theta)
            channel_data.append({"distance": x, "height_diff": delta_y, "raw": d_i})

    return channel_data


def advanced_slope_detection(
    current_frame, frame_history=None, delta_y_thresh=0.005, window=5
):
    """
    高级坡道检测（重构版）：
    状态：平地、检测到有坡、正在上坡
    检测到有坡：输出坡度、坡长
    正在上坡：输出剩余坡长

    Args:
        current_frame (list): 当前帧激光雷达读数
        frame_history (list): 历史帧deltay序列
        delta_y_thresh (float): 判定阈值
        window (int): 滑动窗口大小

    Returns:
        dict: 检测结果，包括状态、坡度、坡长等
    """
    channel_data = process_lidar_frame(current_frame)
    delta_ys = [d["height_diff"] if d else 0 for d in channel_data]
    # 删除delta_ys中等于0的
    delta_ys = [dy for dy in delta_ys if dy != 0]

    # 计算delta_delta_y（相邻通道deltay的差值，用于判断地形变化）
    if len(delta_ys) >= 2:
        delta_delta_ys = np.diff(delta_ys)
    else:
        delta_delta_ys = np.array([])

    # LIDAR统计量
    lidar_valid = [d["raw"] for d in channel_data if d is not None]
    lidar_avg = float(np.mean(lidar_valid)) if lidar_valid else None
    lidar_min = float(np.min(lidar_valid)) if lidar_valid else None
    lidar_max = float(np.max(lidar_valid)) if lidar_valid else None
    lidar_range = (
        float(lidar_max - lidar_min)
        if lidar_valid and lidar_min is not None and lidar_max is not None
        else None
    )
    # 近距离通道（<0.2）、远距离通道（>0.8）统计
    near_channels = [i for i, d in enumerate(channel_data) if d and d["raw"] < 0.2]
    far_channels = [i for i, d in enumerate(channel_data) if d and d["raw"] > 0.8]

    # print(f"delta_delta_ys: {delta_delta_ys}")
    # 状态判断
    status = "未知"
    info = ""
    slope_deg = None
    slope_length = None
    remain_length = None
    if delta_ys and delta_ys[0] > delta_y_thresh:
        status = "正在上坡"
        # 检查是否有deltay开始下降，估算剩余坡长
        # 取最大deltay的通道，若其delta_delta_y为负，说明坡快结束
        if len(delta_delta_ys) > 7 and delta_delta_ys[7] < -0.0005:
            remain_length = round(channel_data[8]["distance"], 2)
            info = f"坡剩余约{remain_length}米"
        else:
            info = "坡上，尚未检测到坡尾"
    elif (
        delta_ys
        and any(dy > delta_y_thresh for dy in delta_ys)
        and len(delta_delta_ys) > 0
        and any(dd > 0.00005 for dd in delta_delta_ys)
    ):
        status = "检测到有坡"
        idxs = [i for i, dy in enumerate(delta_ys) if dy > delta_y_thresh]
        info_list = []
        if len(idxs) >= 2:
            # 用所有满足条件的通道，两两计算角度，最后平均angle为最终坡度
            angles = []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    idx1, idx2 = idxs[i], idxs[j]
                    dx = channel_data[idx2]["distance"] - channel_data[idx1]["distance"]
                    dy = delta_ys[idx2] - delta_ys[idx1]
                    if dx != 0:
                        angle = math.atan2(dy, dx)
                        angles.append(angle)

            if angles:
                # 计算平均角度
                avg_angle = np.mean(angles)
                slope_deg = round(avg_angle * 180 / math.pi, 2)

                # 坡长计算：取最大和最小distance的差值
                idx_min = min(idxs, key=lambda i: channel_data[i]["distance"])
                idx_max = max(idxs, key=lambda i: channel_data[i]["distance"])
                slope_length = round(
                    abs(
                        channel_data[idx_max]["distance"]
                        - channel_data[idx_min]["distance"]
                    ),
                    2,
                )

                info_list.append(f"坡度约{slope_deg}°，坡长约{slope_length}米")
            else:
                info_list.append("坡度信息不足")

            # 额外输出坡起点真实水平距离
            idx_min = min(idxs, key=lambda i: channel_data[i]["distance"])
            theta_min = calculate_theta(idx_min)
            x_min = channel_data[idx_min]["distance"]
            dy_min = channel_data[idx_min]["height_diff"]
            if math.tan(theta_min) != 0:
                x_start = x_min - dy_min / math.tan(theta_min)
                info_list.append(f"坡起点水平距离约{round(x_start, 2)}米")
            else:
                info_list.append("坡起点水平距离计算异常")
        elif len(idxs) == 1:
            # 只有一个通道检测到坡，无法算坡度，只能估算坡起点真实水平距离
            i1 = idxs[0]
            theta1 = calculate_theta(i1)
            x1 = channel_data[i1]["distance"]
            dy1 = channel_data[i1]["height_diff"]
            if math.tan(theta1) != 0:
                x_start = x1 - dy1 / math.tan(theta1)
                info_list.append(f"坡起点水平距离约{round(x_start, 2)}米")
            else:
                info_list.append("坡起点水平距离计算异常")
        else:
            info_list.append("坡信息不足")
        info = "，".join(info_list)
    # 正在上坡：有多组deltay，且delta_delta_y接近0（平台期）
    elif delta_ys and any(abs(dy) < delta_y_thresh for dy in delta_ys):
        status = "平地"
        info = ""
    # 检测到有坡：有一组或多组deltay显著大于0，且delta_delta_y为正
    else:
        status = "未知"
        info = ""

    # 统计量
    avg_delta_y = float(np.mean(delta_ys)) if delta_ys else None
    max_delta_y = float(np.max(delta_ys)) if delta_ys else None
    min_delta_y = float(np.min(delta_ys)) if delta_ys else None
    height_range = (
        float(max_delta_y - min_delta_y)
        if delta_ys and max_delta_y is not None and min_delta_y is not None
        else None
    )
    total_uphill_channels = sum(1 for dy in delta_ys if dy > delta_y_thresh)

    return {
        "status": status,
        "info": info,
        "slope_deg": slope_deg,
        "slope_length": slope_length,
        "remain_length": remain_length,
        "delta_ys": delta_ys,
        "delta_delta_ys": delta_delta_ys.tolist(),
        "frame_history": frame_history,
        # 新增统计量
        "lidar_avg": lidar_avg,
        "lidar_min": lidar_min,
        "lidar_max": lidar_max,
        "lidar_range": lidar_range,
        "near_channels": near_channels,
        "far_channels": far_channels,
        "avg_delta_y": avg_delta_y,
        "max_delta_y": max_delta_y,
        "min_delta_y": min_delta_y,
        "height_range": height_range,
        "total_uphill_channels": total_uphill_channels,
    }


if __name__ == "__main__":
    # 根据Ground Truth添加更多测试帧
    test_frames = [
        (
            30,
            [0.298, 0.301, 0.312, 0.331, 0.361, 0.407, 0.479, 0.599, 0.809, 1.000],
        ),  # 上坡前平地
        (
            35,
            [0.268, 0.271, 0.281, 0.298, 0.325, 0.367, 0.432, 0.539, 0.737, 0.992],
        ),  # ch9首次检测到上坡
        (
            100,
            [0.343, 0.347, 0.359, 0.381, 0.416, 0.466, 0.523, 0.611, 0.754, 1.000],
        ),  # 上坡前
        (
            157,
            [0.374, 0.368, 0.371, 0.383, 0.404, 0.439, 0.492, 0.575, 0.710, 1.000],
        ),  # 开始上坡
        (
            200,
            [0.374, 0.368, 0.371, 0.383, 0.404, 0.439, 0.492, 0.575, 0.710, 1.000],
        ),  # 正在上坡
        (
            250,
            [0.358, 0.353, 0.355, 0.366, 0.387, 0.420, 0.471, 0.551, 0.841, 1.000],
        ),  # 正在上坡
        (
            300,
            [0.354, 0.349, 0.352, 0.363, 0.383, 0.416, 0.507, 0.664, 0.912, 1.000],
        ),  # 正在上坡
        (
            350,
            [0.311, 0.306, 0.312, 0.342, 0.388, 0.447, 0.526, 0.657, 0.903, 1.000],
        ),  # 上坡后期
        (
            365,
            [0.302, 0.305, 0.324, 0.356, 0.398, 0.449, 0.528, 0.660, 0.906, 1.000],
        ),  # 开始下坡
        (
            380,
            [0.320, 0.332, 0.354, 0.384, 0.419, 0.472, 0.556, 0.694, 0.954, 1.000],
        ),  # 下坡中
        (
            404,
            [0.292, 0.297, 0.308, 0.327, 0.356, 0.402, 0.473, 0.591, 0.811, 1.000],
        ),  # 回到平地
    ]

    print("=== 高级坡道检测结果===")
    for step, lidar_readings in test_frames:
        result = advanced_slope_detection(lidar_readings)

        print(f"\nStep {step}:")
        print(f"  状态: {result['status']}")
        print(f"  距离信息: {result['info']}")
        print(f"  坡度: {result['slope_deg']}°, 坡长: {result['slope_length']}米")
        print(f"  剩余坡长: {result['remain_length']}米")
        print(f"  delta_ys: {result['delta_ys']}")
        print(f"  delta_delta_ys: {result['delta_delta_ys']}")
        print(f"  历史deltay序列: {result['frame_history']}")
