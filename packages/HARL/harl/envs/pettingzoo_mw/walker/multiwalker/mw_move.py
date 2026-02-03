from math import exp
import numpy as np
from gymnasium import spaces

from .multiwalker_custom import (
    MultiWalkerEnv as MultiWalkerEnv_base,
    BipedalWalker as BipedalWalker_base,
    TERRAIN_STEP,
    TERRAIN_STARTPAD,
    TERRAIN_HEIGHT,
    LEG_H,
    WALKER_SEPERATION,
    VIEWPORT_W,
    SCALE,
    FPS,
)

MOVE_DOESNT_CARE = 0.8


class BipedalWalker(BipedalWalker_base):
    def __init__(self, *args, **kwargs):
        self.target_v = kwargs.get("target_v", MOVE_DOESNT_CARE)
        self.target_h = kwargs.get("target_h", MOVE_DOESNT_CARE)
        del kwargs["target_v"]
        del kwargs["target_h"]
        super().__init__(*args, **kwargs)

    def get_observation(self):
        original_obs = super().get_observation()

        pos = self.hull.position
        vel = self.hull.linearVelocity
        t_v = self.target_v
        t_h = self.target_h  # if self.target_h != MOVE_DOESNT_CARE else pos[1]
        new_state = original_obs[:14] + [t_v, t_h] + original_obs[14:]

        return new_state

    @property
    def observation_space(self):
        # 24 original obs (joints, etc), 2 displacement obs for each neighboring walker, 3 for package
        original_obs_space = 24 + 4 + 3
        new_obs_space = original_obs_space + 2
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(new_obs_space,),
            dtype=np.float32,
        )


class MultiWalkerEnv(MultiWalkerEnv_base):
    def __init__(self, *args, **kwargs):
        self.reward_factor = kwargs.get("reward_factor", 1.0)
        del kwargs["reward_factor"]
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()

        self.target_v = MOVE_DOESNT_CARE
        self.target_h = MOVE_DOESNT_CARE

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.start_x = [
            init_x + WALKER_SEPERATION * i * TERRAIN_STEP for i in range(self.n_walkers)
        ]
        self.walkers = [
            BipedalWalker(
                self,
                self.world,
                init_x=sx,
                init_y=init_y,
                seed=self.seed_val,
                target_v=self.target_v,
                target_h=self.target_h,
            )
            for sx in self.start_x
        ]
        self.observation_space = [agent.observation_space for agent in self.walkers]
        self.action_space = [agent.action_space for agent in self.walkers]
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(
                self.n_walkers * (24 + 2) + 3,
            ),  # 24 is the observation space of each walker, 3 is the package observation space
            dtype=np.float32,
        )

    def reset(self):
        super().reset()
        self.set_target(MOVE_DOESNT_CARE, MOVE_DOESNT_CARE)
        return self.observe(0)

    def set_t_v_agent(self, agent_idx: int, target_v: float):
        self.walkers[agent_idx].target_v = target_v

    def set_t_h_agent(self, agent_idx: int, target_h: float):
        self.walkers[agent_idx].target_h = target_h

    def set_target_v(self, target_v):
        self.target_v = target_v
        for i in range(self.n_walkers):
            self.set_t_v_agent(i, target_v)

    def set_target_h(self, target_h):
        self.target_h = target_h
        for i in range(self.n_walkers):
            self.set_t_h_agent(i, target_h)

    def set_target(self, target_v, target_h):
        self.set_target_v(target_v)
        self.set_target_h(target_h)

    def get_target_v_agent(self, agent_idx: int) -> float:
        return self.walkers[agent_idx].target_v

    def get_target_h_agent(self, agent_idx: int) -> float:
        return self.walkers[agent_idx].target_h

    def scroll_subroutine(self):
        rewards, done, obs = super().scroll_subroutine()

        def _calc_bowl(ref_point: float, cur_point: float) -> float:
            sigma = 0.093  
            return exp(-((ref_point - cur_point) ** 2) / (2 * sigma**2))

        for i in range(self.n_walkers):
            if self.walkers[i].hull is None:
                continue
            # v_deviation_penalty
            v_x = (
                0.3 * self.walkers[i].hull.linearVelocity.x * (VIEWPORT_W / SCALE) / FPS
            )
            reward_v_deviation_penalty = self.reward_factor * _calc_bowl(
                self.get_target_v_agent(i), v_x
            )
            rewards[i] += reward_v_deviation_penalty

            # h_deviation_penalty
            # pos = self.walkers[i].hull.position[1]
            # reward_h_deviation_penalty = -self.reward_factor * abs(
            #     self.target_h - pos
            # ) 
            # rewards[i] += reward_h_deviation_penalty

        return rewards, done, obs
