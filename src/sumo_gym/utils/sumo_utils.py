import os
import sys
import traci
import traci.constants as tc

import sumo_gym.typing


# SUMO_COMMAND = ["/opt/homebrew/opt/sumo/share/sumo/bin/sumo-gui", "-c", "assets/data/sumo_simulation.sumocfg"]
# SUMO_COMMAND = [
#     "/usr/local/Cellar/sumo/1.10.0/bin/sumo-gui",
#     "-c",
#     "assets/data/sumo_simulation.sumocfg",
# ]

# TODO: fake the data, we need to exclude the turnaround from MDP action space, and make render function parameterless
# below is a test route
TEST_LOCATIONS = [14, 15, 2, 3, 4, 7, 12, 13]
LANE_LENGTH = [23.42, 22.60, 19.98, 4.43, 2.20, 17.31, 49.12]


class SumoRender:
    def __init__(
        self,
        sumo_gui_path: str = None,
        vertex_dict: dict = None,
        edge_dict: dict = None,
        ev_dict: dict = None,
        edges: sumo_gym.typing.EdgeType = None,
    ):
        self.sumo_gui_path = sumo_gui_path
        self.vertex_dict = vertex_dict
        self.edge_dict = edge_dict
        self.ev_dict = ev_dict
        self.edges = edges
        self.initialized = False
        self.terminated = False
        self.ev_stop = [False]
        self.count = 2  # 0,1 is the first edge, must go
        self.routes = [()]

        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        traci.start([self.sumo_gui_path, "-c", "assets/data/sumo_simulation.sumocfg"])

    def render(self):

        while not self.terminated:

            if not self.initialized:
                print("Initialize the first starting edge for vehicle...")
                self.initialized = True
                
                # TODO: change 1 to vehicle count
                for i in range(1):

                    vehicle_id = self._find_key_from_value(self.ev_dict, i)

                    via_edge = (TEST_LOCATIONS[0], TEST_LOCATIONS[1])
                    edge_id = self._find_key_from_value(
                        self.edge_dict, self._find_edge_index(via_edge)
                    )
                    self.routes[i] = tuple([edge_id])

                    # stop at the ending vertex of vehicle's starting edge
                    # notice here each vehicle must finish traveling along it starting edge
                    # there is no way to reassign it.
                    print("Step stop for vehicle: ", vehicle_id)
                    traci.vehicle.setStop(
                        vehID=vehicle_id,
                        edgeID=edge_id,
                        pos=LANE_LENGTH[0],
                        laneIndex=0,
                        duration=189999999999,
                        flags=0,
                        startPos=0,
                    )

            # change 'from', edge accordingly
            for i in range(1):
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                if self.ev_stop[i]:
                    self.ev_stop[i] = False
                    vehicle_id = self._find_key_from_value(self.ev_dict, i)
                    via_edge = (TEST_LOCATIONS[self.count - 1], TEST_LOCATIONS[self.count])
                    
                    edge_id = self._find_key_from_value(
                        self.edge_dict, self._find_edge_index(via_edge)
                    )
                    self.routes[i] += tuple([edge_id])
                    
                    print("Vehicle ", vehicle_id," has arrived stopped location, reassign new routes: ", self.routes[i])
                    
                    traci.vehicle.setRoute(vehID=vehicle_id, edgeList=self.routes[i][-2:])
                    traci.vehicle.setStop(
                        vehID=vehicle_id,
                        edgeID=edge_id,
                        pos=LANE_LENGTH[self.count-1],
                        laneIndex=0,
                        duration=189999999999,
                        flags=0,
                        startPos=0,
                    )
                    self.count += 1
            
            traci.simulationStep()

            for i in range(1):
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                if traci.vehicle.getStopState(vehicle_id) == 1:  # arrived the assigned vertex, can be assigned to the next
                    self.terminated = self.count == len(TEST_LOCATIONS)
                    self.ev_stop[i] = True
                    traci.vehicle.resume(vehicle_id)


    def close(self):
        traci.close()

    def _find_key_from_value(self, dict, value):
        return list(dict.keys())[list(dict.values()).index(value)]

    def _find_edge_index(self, via_edge):
        for i in range(len(self.edges)):
            if via_edge[0] == self.edges[i].start and via_edge[1] == self.edges[i].end:
                return i
