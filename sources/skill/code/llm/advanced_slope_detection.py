import math
import numpy as np


def calculate_theta(i):
    return math.pi / 2 - 0.15 * i


def process_lidar_frame(lidar_readings):

    h = lidar_readings[0]
    channel_data = []

    for i in range(10):
        d_i = lidar_readings[i]
        if d_i >= 1.0: 
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

    channel_data = process_lidar_frame(current_frame)
    delta_ys = [d["height_diff"] if d else 0 for d in channel_data]

    delta_ys = [dy for dy in delta_ys if dy != 0]

    if len(delta_ys) >= 2:
        delta_delta_ys = np.diff(delta_ys)
    else:
        delta_delta_ys = np.array([])

    lidar_valid = [d["raw"] for d in channel_data if d is not None]
    lidar_avg = float(np.mean(lidar_valid)) if lidar_valid else None
    lidar_min = float(np.min(lidar_valid)) if lidar_valid else None
    lidar_max = float(np.max(lidar_valid)) if lidar_valid else None
    lidar_range = (
        float(lidar_max - lidar_min)
        if lidar_valid and lidar_min is not None and lidar_max is not None
        else None
    )

    near_channels = [i for i, d in enumerate(channel_data) if d and d["raw"] < 0.2]
    far_channels = [i for i, d in enumerate(channel_data) if d and d["raw"] > 0.8]

    # print(f"delta_delta_ys: {delta_delta_ys}")

    status = "unknown"
    info = ""
    slope_deg = None
    slope_length = None
    remain_length = None
    if delta_ys and delta_ys[0] > delta_y_thresh:
        status = "slope"

        if len(delta_delta_ys) > 7 and delta_delta_ys[7] < -0.0005:
            remain_length = round(channel_data[8]["distance"], 2)
            info = f"remaining{remain_length}m"
        else:
            info = "slope, not detected tail"
    elif (
        delta_ys
        and any(dy > delta_y_thresh for dy in delta_ys)
        and len(delta_delta_ys) > 0
        and any(dd > 0.00005 for dd in delta_delta_ys)
    ):
        status = "slope detected"
        idxs = [i for i, dy in enumerate(delta_ys) if dy > delta_y_thresh]
        info_list = []
        if len(idxs) >= 2:

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

                avg_angle = np.mean(angles)
                slope_deg = round(avg_angle * 180 / math.pi, 2)

                idx_min = min(idxs, key=lambda i: channel_data[i]["distance"])
                idx_max = max(idxs, key=lambda i: channel_data[i]["distance"])
                slope_length = round(
                    abs(
                        channel_data[idx_max]["distance"]
                        - channel_data[idx_min]["distance"]
                    ),
                    2,
                )

                info_list.append(f"slope{slope_deg}°, slope{slope_length}m")
            else:
                info_list.append("slope info not enough")

            idx_min = min(idxs, key=lambda i: channel_data[i]["distance"])
            theta_min = calculate_theta(idx_min)
            x_min = channel_data[idx_min]["distance"]
            dy_min = channel_data[idx_min]["height_diff"]
            if math.tan(theta_min) != 0:
                x_start = x_min - dy_min / math.tan(theta_min)
                info_list.append(f"start{round(x_start, 2)}m")
            else:
                info_list.append("start calculation error")
        elif len(idxs) == 1:
            i1 = idxs[0]
            theta1 = calculate_theta(i1)
            x1 = channel_data[i1]["distance"]
            dy1 = channel_data[i1]["height_diff"]
            if math.tan(theta1) != 0:
                x_start = x1 - dy1 / math.tan(theta1)
                info_list.append(f"start{round(x_start, 2)}m")
            else:
                info_list.append("start calculation error")
        else:
            info_list.append("slope info not enough")
        info = "，".join(info_list)
    elif delta_ys and any(abs(dy) < delta_y_thresh for dy in delta_ys):
        status = "flat"
        info = ""
    else:
        status = "unknown"
        info = ""

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
    test_frames = [
        (
            30,
            [0.298, 0.301, 0.312, 0.331, 0.361, 0.407, 0.479, 0.599, 0.809, 1.000],
        ),  
        (
            35,
            [0.268, 0.271, 0.281, 0.298, 0.325, 0.367, 0.432, 0.539, 0.737, 0.992],
        ),  
        (
            100,
            [0.343, 0.347, 0.359, 0.381, 0.416, 0.466, 0.523, 0.611, 0.754, 1.000],
        ),  
        (
            157,
            [0.374, 0.368, 0.371, 0.383, 0.404, 0.439, 0.492, 0.575, 0.710, 1.000],
        ),  
        (
            200,
            [0.374, 0.368, 0.371, 0.383, 0.404, 0.439, 0.492, 0.575, 0.710, 1.000],
        ),  
        (
            250,
            [0.358, 0.353, 0.355, 0.366, 0.387, 0.420, 0.471, 0.551, 0.841, 1.000],
        ),  
        (
            300,
            [0.354, 0.349, 0.352, 0.363, 0.383, 0.416, 0.507, 0.664, 0.912, 1.000],
        ),  
        (
            350,
            [0.311, 0.306, 0.312, 0.342, 0.388, 0.447, 0.526, 0.657, 0.903, 1.000],
        ),  
        (
            365,
            [0.302, 0.305, 0.324, 0.356, 0.398, 0.449, 0.528, 0.660, 0.906, 1.000],
        ),  
        (
            380,
            [0.320, 0.332, 0.354, 0.384, 0.419, 0.472, 0.556, 0.694, 0.954, 1.000],
        ),  
        (
            404,
            [0.292, 0.297, 0.308, 0.327, 0.356, 0.402, 0.473, 0.591, 0.811, 1.000],
        ),  
    ]

    print("=== advanced slope detection result===")
    for step, lidar_readings in test_frames:
        result = advanced_slope_detection(lidar_readings)

        print(f"\nStep {step}:")
        print(f"  status: {result['status']}")
        print(f"  distance info: {result['info']}")
        print(f"  slope: {result['slope_deg']}°, slope length: {result['slope_length']}m")
        print(f"  remaining length: {result['remain_length']}m")
        print(f"  delta_ys: {result['delta_ys']}")
        print(f"  delta_delta_ys: {result['delta_delta_ys']}")
        print(f"  history delta_y sequence: {result['frame_history']}")
