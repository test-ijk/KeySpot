# from mapdn.environments.var_voltage_control.disturbances import DisturbanceBase
# from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl


class LoadChange:
    """
    对负荷做扰动的类。

    disturbance_args: dict = {"multiplier": 2}
    """

    def __init__(self, env, disturbance_args: dict):
        # super().__init__(env, disturbance_args)
        self.type = "load_change"
        self.env = env
        self.disturbance_args = disturbance_args

    def start(self):
        self.env = self.env  # 激活python类型推断

        # update the record in the pandapower
        # self.env.powergrid.sgen["p_mw"] = self.env.powergrid.sgen["p_mw"] * self.disturbance_args["multiplier"]
        # self.env.powergrid.load["p_mw"] = (
        #     self.env.powergrid.load["p_mw"] * self.disturbance_args["multiplier"]
        # )

        self.env.powergrid.load["q_mvar"] = (
            self.env.powergrid.load["q_mvar"] * self.disturbance_args["multiplier"]
        )
        # self.env.powergrid.sgen["p_mw"] = (
        #     self.env.powergrid.sgen["p_mw"] * self.disturbance_args["multiplier"]
        # )
        # for i, v in enumerate(self.env.s_max):
        #     if self.env.powergrid.sgen["p_mw"][i] > v:
        #         self.env.powergrid.sgen["p_mw"][i] = v

        # rich.print(self.env.powergrid.load["q_mvar"])

    def end(self):
        pass
