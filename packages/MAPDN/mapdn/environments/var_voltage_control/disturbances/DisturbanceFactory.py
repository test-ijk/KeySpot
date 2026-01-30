# from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn.environments.var_voltage_control.disturbances import DisturbanceBase
from mapdn.environments.var_voltage_control.disturbances.load.change import LoadChange
from mapdn.environments.var_voltage_control.disturbances.pv.stop import PVStop
from mapdn.environments.var_voltage_control.disturbances.DisturbanceConfig import (
    DisturbanceConfig,
)
import numpy as np

disturbance_dict = {
    "load_change": LoadChange,
    "pv_stop": PVStop,
}


class DisturbanceFactory:
    def __init__(
        self,
        config: DisturbanceConfig,
        base_env,
    ):
        self.env = base_env
        self.disturbance: DisturbanceBase = disturbance_dict[config.type](
            env=self.env,
            disturbance_args=config.disturbance_args,
        )
        self.disturbance_args = config.disturbance_args
        self.is_random = config.is_random
        if self.is_random:
            self.random_probability = config.random_probability
            self.random_duration = config.random_duration
            self.start_at = 1e9
            self.end_at = 1e9
        else:
            self.random_probability = 0
            self.random_duration = 0
            self.start_at = config.start_at
            self.end_at = config.end_at

        self.random_generator = np.random.RandomState(42)

        self.random_is_active = False
        self.random_start_at = None
        self.random_should_end_at = None

        self.should_trigger = False

    def execute_with_frame(self, frame: int):
        if self.is_random:
            if not self.random_is_active:
                rand_it = self.random_generator.rand()
                # print(f"{rand_it} | {self.random_probability}")
                if rand_it < self.random_probability:
                    print(f"Random disturbance is triggered at frame {frame}")
                    self.should_trigger = True
                    self.random_is_active = True
                    self.random_start_at = frame
                    self.random_should_end_at = frame + self.random_duration
            else:
                if frame >= self.random_should_end_at:
                    self.should_trigger = False
                    self.random_is_active = False
                    self.random_start_at = None
                    self.random_should_end_at = None
        else:
            if frame >= self.start_at:
                self.should_trigger = True
            elif frame >= self.end_at:
                self.should_trigger = False

        if self.should_trigger:
            self.start()
        else:
            self.recover()

    def reset(self):
        self.random_is_active = False
        self.random_start_at = None
        self.random_should_end_at = None

    def start(self):
        self.disturbance.start()

    def recover(self):
        self.disturbance.end()
