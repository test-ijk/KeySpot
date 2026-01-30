from harl.envs.pettingzoo_mw.walker.disturbances import DisturbanceBase, MultiWalkerEnv


class SkillMoveBase(DisturbanceBase):


    def __init__(self, env: MultiWalkerEnv, disturbance_args: dict):
        super().__init__(env, disturbance_args)
        self.color1 = (255, 0, 0)
        self.color2 = (255, 0, 0)

    def colorize(self, color1: tuple = (255, 0, 0), color2: tuple = (255, 0, 0)):
        for agent_id in self.disturbance_args.get(
            "effect_on_agent", range(self.env.n_walkers)
        ):
            self.env.agents[agent_id].hull.color1 = color1
            self.env.agents[agent_id].hull.color2 = color2

    def un_colorize(self):
        for agent_id in self.disturbance_args.get(
            "effect_on_agent", range(self.env.n_walkers)
        ):
            self.env.agents[agent_id].hull.color1 = (127, 51, 229)
            self.env.agents[agent_id].hull.color2 = (76, 76, 127)

    def start(self):
        self.colorize()
        pass

    def end(self):
        self.un_colorize()
        pass
