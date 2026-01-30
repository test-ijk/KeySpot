from typing import Any

from .sumo_event_manager import SumoEventManager


class LaneCloseEventManager(SumoEventManager):
    def _event_start(self, args: Any) -> None:
        print("[LaneCloseEventManager] starting!!!")
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        lane_id = "A2B2_1"
        self.sumo_env.lane.setDisallowed(lane_id, ["all"])

        vehicle_ids = self.sumo_env.vehicle.getIDList() 
        for vehId in vehicle_ids:
            route = self.sumo_env.vehicle.getRoute(
                vehId
            )  
            if "A2B2" in route:
                waiting_time = self.sumo_env.vehicle.getWaitingTime(
                    vehId
                ) 

                if waiting_time > 100:
                    print(
                        f"Removed vehicle {vehId} because route contains edge {lane_id}"
                    )
                    self.sumo_env.vehicle.remove(vehId)
                else:
                    self.sumo_env.vehicle.rerouteEffort(vehId)

    def _event_stop(self) -> None:
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        self.sumo_env.lane.setDisallowed("A2B2_1", [])

        vehicle_ids = self.sumo_env.vehicle.getIDList()  
        for vehId in vehicle_ids:
            self.sumo_env.vehicle.rerouteEffort(vehId)

    def _event_random_value(self) -> Any:
        pass
