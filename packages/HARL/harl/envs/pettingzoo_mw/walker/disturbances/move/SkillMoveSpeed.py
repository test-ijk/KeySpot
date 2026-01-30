from harl.envs.pettingzoo_mw.walker.disturbances.move import SkillMoveBase
from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import MOVE_DOESNT_CARE


class SkillMoveSpeed(SkillMoveBase):
    """

    disturbance_args: dict = {"mass": 4.57}
    """

    def start(self):
        super().start()
        if self.disturbance_args.get("on_agent_idx") is not None:
            self.env.set_t_v_agent(
                self.disturbance_args["on_agent_idx"], self.disturbance_args["speed"]
            )
        else:
            self.env.set_target_v(self.disturbance_args["speed"])

    def end(self):
        if self.disturbance_args.get("on_agent_idx") is not None:
            self.env.set_t_v_agent(
                self.disturbance_args["on_agent_idx"], MOVE_DOESNT_CARE
            )
        else:
            self.env.set_target_v(MOVE_DOESNT_CARE)
        super().end()
