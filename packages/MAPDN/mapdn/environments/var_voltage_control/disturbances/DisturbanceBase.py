# from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl


class DisturbanceBase:
    """
    所有扰动类的基类。
    存储了环境和扰动变量。
    """

    def __init__(self, env, disturbance_args: dict):
        self.env = env
        self.disturbance_args = disturbance_args
        self.type = "undefined"

    def start(self):
        pass

    def end(self):
        pass
