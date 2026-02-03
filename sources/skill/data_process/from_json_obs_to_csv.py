import json
import numpy as np
import csv
import os
import pandas as pd

position_prefix = ""
json_position = f"{position_prefix}_episode_obses.json"
lidar_json_position = f"{position_prefix}_lidar_obs.json"


def get_observation_labels(obs_dim):

    if obs_dim == 24:

        return (
            [
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
            ]
            + [f"lidar_{i}" for i in range(10)]
        ) 

    elif obs_dim == 26:

        return [
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
            "speed_factor_hip",
            "speed_factor_knee", 
        ] + [f"lidar_{i}" for i in range(10)]  

    elif obs_dim == 31:

        base_labels = get_observation_labels(26 if obs_dim > 26 else 24)
        neighbor_labels = [
            "left_neighbor_x_offset",  
            "left_neighbor_y_offset",  
            "right_neighbor_x_offset", 
            "right_neighbor_y_offset",  
            "package_x_offset",
            "package_y_offset", 
            "package_angle",  
        ]
        return base_labels + neighbor_labels

    elif obs_dim == 33:

        return [
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
            "speed_factor_hip",
            "speed_factor_knee",  
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

    else:

        return [f"obs_{i}" for i in range(obs_dim)]


def analyze_observation_at_75_percent(data):

    if not data or not data[0] or not data[0][0]:
        print("")
        return


    episode = data[0]
    n_agents = len(episode)


    for agent_idx in range(n_agents):
        agent_obs = episode[agent_idx]
        total_steps = len(agent_obs)


        step_75_percent = int(total_steps * 0.5)

        if step_75_percent < total_steps:
            obs_75_raw = agent_obs[step_75_percent]
            print(f"debug info: Agent {agent_idx} original observation type: {type(obs_75_raw)}")
            print(
                f"debug info: original observation length: {len(obs_75_raw) if hasattr(obs_75_raw, '__len__') else 0}"
            )

            if (
                isinstance(obs_75_raw, list)
                and len(obs_75_raw) == 1
                and isinstance(obs_75_raw[0], list)
            ):
                obs_75 = obs_75_raw[0]  
                print("debug info: nested structure detected, expanded")
            else:
                obs_75 = obs_75_raw

            obs_dim = len(obs_75) if hasattr(obs_75, "__len__") else 0
            print(f"debug info: actual observation dimension: {obs_dim}")
            if obs_dim > 0:
                print(
                    f"debug info: actual observation example (first 5): {obs_75[:5] if obs_dim >= 5 else obs_75}"
                )
            labels = get_observation_labels(obs_dim)

            print(
                f"\nAgent {agent_idx} - 75% position observation (step {step_75_percent}/{total_steps}):"
            )
            print(f"observation dimension: {obs_dim}")

            for i, (label, value) in enumerate(zip(labels, obs_75)):
                if not isinstance(label, str):
                    label = str(label)
                try:
                    value_str = f"{float(value):8.4f}"
                except (ValueError, TypeError):
                    value_str = f"{str(value):>8s}"
                print(f"  [{i:2d}] {label:25s}: {value_str}")

            if obs_dim >= 14:
                print(
                    f"    hull angle: {obs_75[0]:6.3f} rad ({np.degrees(obs_75[0]):6.1f}°)"
                )
                print(f"    x direction speed: {obs_75[2]:6.3f}")
                print(f"    y direction speed: {obs_75[3]:6.3f}")
                print(f"    left foot contact: {'yes' if obs_75[8] > 0.5 else 'no'}")
                print(f"    right foot contact: {'yes' if obs_75[13] > 0.5 else 'no'}")

                if obs_dim >= 26:  
                    print(f"    hip speed factor: {obs_75[14]:6.3f}")
                    print(f"    knee speed factor: {obs_75[15]:6.3f}")

                lidar_start = 16 if obs_dim >= 26 else 14
                if obs_dim > lidar_start:
                    lidar_data = obs_75[lidar_start : lidar_start + 10]
                    print(f"    lidar average distance: {np.mean(lidar_data):6.3f}")
                    print(f"    lidar minimum distance: {np.min(lidar_data):6.3f}")


def compare_observations_across_time(data):
    if not data or not data[0] or not data[0][0]:
        print("warning: data is incomplete, skip time comparison analysis")
        return

    episode = data[0]
    agent_0_obs = episode[0]
    total_steps = len(agent_0_obs)

    if total_steps > 0:
        first_obs_raw = agent_0_obs[0]
        print(f"\ndebug info: Agent 0 first observation type: {type(first_obs_raw)}")
        print(
            f"debug info: original length: {len(first_obs_raw) if hasattr(first_obs_raw, '__len__') else 'N/A'}"
        )

        if (
            isinstance(first_obs_raw, list)
            and len(first_obs_raw) == 1
            and isinstance(first_obs_raw[0], list)
        ):
            print("debug info: nested structure detected, will be expanded in comparison")
        elif hasattr(first_obs_raw, "__len__") and len(first_obs_raw) > 0:
            print(
                f"debug info: first observation example: {first_obs_raw[:5] if len(first_obs_raw) >= 5 else first_obs_raw}"
            )

    time_points = [
        int(total_steps * 0.25),  # 25%
        int(total_steps * 0.50),  # 50%
        int(total_steps * 0.75),  # 75%
        total_steps - 1,  # 
    ]

    print("\n=== comparison of observations at different time points (Agent 0) ===")

    key_indices = [0, 2, 3, 8, 13]  # hull angle, x speed, y speed, left foot contact, right foot contact
    key_labels = ["hull angle", "x speed", "y speed", "left foot contact", "right foot contact"]

    print(f"{'time point':>10s}", end="")
    for label in key_labels:
        print(f"{label:>12s}", end="")
    print()

    for i, step in enumerate(time_points):
        if step < total_steps:
            obs_raw = agent_0_obs[step]

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
                        if idx in [8, 13]:  # contact flag
                            result = "yes" if float(value) > 0.5 else "no"
                            print(f"{result:>12s}", end="")
                        else:
                            print(f"{float(value):>12.4f}", end="")
                    except (ValueError, TypeError, IndexError):
                        print(f"{str(obs[idx]):>12s}", end="")
                else:
                    print(f"{'N/A':>12s}", end="")
            print()


def extract_agents_lidar_to_csv(
    data, agent_indices=[1, 2, 3], output_dir="results/obs_csv"
):
    os.makedirs(output_dir, exist_ok=True)

    if not data or len(data) == 0:
        print("error: data is empty")
        return

    for agent_idx in agent_indices:
        if len(data[0]) <= agent_idx:
            print(f"error: data does not have agent {agent_idx}")
            continue

        agent_obs = data[0][agent_idx]

        first_obs = agent_obs[0]
        if (
            isinstance(first_obs, list)
            and len(first_obs) == 1
            and isinstance(first_obs[0], list)
        ):
            first_obs = first_obs[0]

        obs_dim = len(first_obs)
        lidar_start = 16 if obs_dim >= 26 else 14  # determine lidar start index based on observation dimension

        output_file = os.path.join(output_dir, f"agent_{agent_idx}_lidar_data.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + [f"lidar_{i}" for i in range(10)]
            writer.writerow(header)

            for step, obs in enumerate(agent_obs):
                if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], list):
                    obs = obs[0]

                lidar_data = obs[lidar_start : lidar_start + 10]
                row = [step] + lidar_data
                writer.writerow(row)

        print(f"successfully saved lidar data for agent {agent_idx} to {output_file}")


def extract_all_observations_to_csv(data, output_dir="results/obs_csv"):

    os.makedirs(output_dir, exist_ok=True)

    if not data or len(data) == 0:
        print("error: data is empty")
        return

    episode = data[0]
    n_agents = len(episode)

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

        output_file = os.path.join(output_dir, f"agent_{agent_idx}_observations.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + labels
            writer.writerow(header)

            for step, obs in enumerate(agent_obs):
                if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], list):
                    obs = obs[0]
                row = [step] + obs
                writer.writerow(row)

        print(f"successfully saved all observations for agent {agent_idx} to {output_file}")


print(f"reading file: {json_position}")
try:
    with open(json_position, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("file read successfully")
except FileNotFoundError:
    print(f"error: file not found - {json_position}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"error: JSON parsing failed - {e}")
    exit(1)

print("=== data structure dimension analysis ===")
print(f"overall structure type: {type(data)}")

if not isinstance(data, list):
    print(f"error: data should be a list type, but is: {type(data)}")
    exit(1)

if len(data) == 0:
    print("error: data is empty")
    exit(1)

print(f"1st layer - number of episodes: {len(data)}")

if len(data) > 0:
    first_episode = data[0]
    print(f"2nd layer - number of agents in each episode: {len(first_episode)}")

    if len(first_episode) > 0:
        first_agent = first_episode[0]
        print(f"3rd layer - number of observations for each agent: {len(first_agent)}")

        if len(first_agent) > 0:
            first_obs_raw = first_agent[0]
            print(f"4th layer - original observation data type: {type(first_obs_raw)}")
            print(
                f"4th layer - original observation length: {len(first_obs_raw) if hasattr(first_obs_raw, '__len__') else 0}"
            )

            if (
                isinstance(first_obs_raw, list)
                and len(first_obs_raw) == 1
                and isinstance(first_obs_raw[0], list)
            ):
                first_obs = first_obs_raw[0]
                print("nested structure detected, expanded")
            else:
                first_obs = first_obs_raw

            obs_dim = len(first_obs) if hasattr(first_obs, "__len__") else 0
            print(f"4th layer - actual observation dimension: {obs_dim}")

            labels = get_observation_labels(obs_dim)
            print(f"\n=== observation dimension description (total {obs_dim} dimensions) ===")
            if obs_dim > 0:
                print(
                    f"observation data example (first 10 values): {first_obs[:10] if obs_dim >= 10 else first_obs}"
                )

                if not isinstance(labels, list):
                    print(f"error: labels is not a list type, but: {type(labels)}")
                    labels = [f"obs_{i}" for i in range(obs_dim)]

                if len(labels) != obs_dim:
                    print(f"warning: number of labels ({len(labels)}) does not match the observation dimension ({obs_dim})")
                    labels = [f"obs_{i}" for i in range(obs_dim)]

                for i, (label, value) in enumerate(zip(labels, first_obs)):
                    if not isinstance(label, str):
                        label = str(label)
                    try:
                        value_str = f"{float(value):8.4f}"
                    except (ValueError, TypeError):
                        value_str = f"{str(value):>8s}"
                    print(f"  [{i:2d}] {label:25s}: {value_str}")

print("\n=== detailed dimension statistics ===")
for episode_idx, episode in enumerate(data):
    print(f"Episode {episode_idx}: {len(episode)} agents")
    for agent_idx, agent_obs in enumerate(episode):
        print(f"  Agent {agent_idx}: {len(agent_obs)} observations")
        if len(agent_obs) > 0:
            obs_dim = len(agent_obs[0])
            print(f"    each observation dimension: {obs_dim}")
    if episode_idx >= 2:
        print("  ... (more episodes)")
        break

print("\n=== summary ===")
print("data structure: [Episodes][Agents][TimeSteps][ObsDimension]")
print(
    f"specific dimension: [{len(data)}][{len(data[0]) if data else 0}][variable length][{len(data[0][0][0]) if data and data[0] and data[0][0] else 0}]"
)

extract_all_observations_to_csv(data)

analyze_observation_at_75_percent(data)

compare_observations_across_time(data)

extract_agents_lidar_to_csv(data, agent_indices=[0, 1, 2], output_dir="results/obs_csv")

with open(lidar_json_position, "r", encoding="utf-8") as f:
    lidar_data = json.load(f)

for agent_idx in range(3):
    rows = []
    for step_idx, step_obs in enumerate(lidar_data[0][agent_idx]):
        print(len(step_obs))
        row = {"episode": 0, "step": step_idx}
        for i in range(10):
            row[f"lidar_{i}"] = step_obs[agent_idx][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = lidar_json_position.replace(".json", f"_agent{agent_idx}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Agent {agent_idx} CSV file saved to: {csv_path}")
    print(df.head())
