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
)


class BipedalWalker(BipedalWalker_base):
    def get_observation(self):
        original_obs = super().get_observation()

        new_state = (
            original_obs[:14]
            + [self.env.terrain[0].fixtures[0].friction]
            + original_obs[14:]
        )

        return new_state

    @property
    def observation_space(self):
        # 24 original obs (joints, etc), 2 displacement obs for each neighboring walker, 3 for package
        original_obs_space = 24 + 4 + 3
        new_obs_space = original_obs_space + 1  # 添加重力信息
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(new_obs_space,),
            dtype=np.float32,
        )


class MultiWalkerEnv(MultiWalkerEnv_base):
    def setup(self):
        super().setup()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.start_x = [
            init_x + WALKER_SEPERATION * i * TERRAIN_STEP for i in range(self.n_walkers)
        ]
        self.walkers = [
            BipedalWalker(
                self, self.world, init_x=sx, init_y=init_y, seed=self.seed_val
            )
            for sx in self.start_x
        ]
        self.observation_space = [agent.observation_space for agent in self.walkers]
        self.action_space = [agent.action_space for agent in self.walkers]
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(
                self.n_walkers * (24 + 1) + 3,
            ),  # 24 is the observation space of each walker, 3 is the package observation space
            dtype=np.float32,
        )
