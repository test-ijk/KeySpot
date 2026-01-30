def mapdn_customize_dict(cfg, algo_dict: dict, env_dict: dict, save_group: str):
    print("env_dict")
    print(env_dict)
    if algo_dict["train"].get("episode_length") is not None:
        max_cycles = env_dict["max_cycles"]
        algo_dict["train"]["episode_length"] = max_cycles - 1
        print(
            f"algo_dict['train']['episode_length'] = {algo_dict['train']['episode_length']}"
        )
    scenario = env_dict["scenario"]
    if scenario == "case33_3min_final":
        env_dict["action_bias"] = 0.0
        env_dict["action_scale"] = 0.8
    elif scenario == "case141_3min_final":
        env_dict["action_bias"] = 0.0
        env_dict["action_scale"] = 0.6
    elif scenario == "case322_3min_final":
        env_dict["action_bias"] = 0.0
        env_dict["action_scale"] = 0.8
    env_dict["data_path"] = f"{env_dict['data_path']}{scenario}"
    return algo_dict, env_dict


__all__ = ["mapdn_customize_dict"]
