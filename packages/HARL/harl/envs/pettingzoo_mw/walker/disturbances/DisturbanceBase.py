from harl.envs.pettingzoo_mw.walker.multiwalker.multiwalker_base import MultiWalkerEnv


class DisturbanceBase:


    def __init__(self, env: MultiWalkerEnv, disturbance_args: dict):
        self.env = env
        self.disturbance_args = disturbance_args

    def start(self):
        pass

    def end(self):
        pass
