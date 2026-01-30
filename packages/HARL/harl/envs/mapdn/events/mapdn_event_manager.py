from typing import Any
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from ..mapdn_env import MapdnWrapperEnv
from ...harl_env_with_events import EventManager


class MapdnEventManager(EventManager[MapdnWrapperEnv, VoltageControl]):
    def _extract_real_env(self) -> VoltageControl:
        self.real_env: VoltageControl = self.env.real_env
        assert self.real_env is not None
        return self.real_env

    def _event_start(self, args: Any) -> None:
        self.real_env = self._extract_real_env()
        assert self.real_env is not None
        pass

    def _event_stop(self) -> None:
        self.real_env = self._extract_real_env()
        assert self.real_env is not None
        pass
