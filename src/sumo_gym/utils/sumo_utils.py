import os
import sys
import traci

import sumo_gym.typing
from sumo_gym.utils.fmp_utils import IDLE_LOCATION


STOPPED_STATUS = 1


class SumoRender:
    def __init__(
        self,
        sumo_gui_path: str = None,
        sumo_config_path: str = None,
        edge_dict: dict = None,
        edge_length_dict: dict = None,
        ev_dict: dict = None,
        edges: sumo_gym.typing.EdgeType = None,
        n_vehicle: int = 1,
    ):
        self.sumo_gui_path = sumo_gui_path
        self.edge_dict = edge_dict
        self.edge_length_dict = edge_length_dict
        self.ev_dict = ev_dict
        self.edges = edges
        self.initialized = False
        self.terminated = False
        self.stop_statuses = [False] * n_vehicle
        self.n_vehicle = n_vehicle
        self.routes = []

        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        traci.start([self.sumo_gui_path, "-c", sumo_config_path])

    def get_stop_status(self):
        return self.stop_statuses

    def update_travel_vertex_info_for_vehicle(self, vehicle_travel_info_list):
        self.travel_info = vehicle_travel_info_list

    def render(self):

        # while not self.terminated:
        if not self.initialized:
            print("Initialize the first starting edge for vehicle...")
            self.initialized = True
            self._initialize_route()
            self._park_vehicle_to_assigned_starting_pos()
        else:
            self._update_route_with_stop()
            traci.simulationStep()
            self._update_stop_status()

    def close(self):
        while not self.terminated:
            traci.simulationStep()

            self.terminated = True

            eligible_vehicle = traci.vehicle.getIDList()
            for i in range(self.n_vehicle):
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                if vehicle_id not in eligible_vehicle:
                    continue

                if (
                    traci.vehicle.getStopState(vehicle_id) != STOPPED_STATUS
                ):  # as long as one vehicle not arrive its assigned last vertex, continue simulation
                    self.terminated = False

        traci.close()

    def _initialize_route(self):
        for i in range(self.n_vehicle):

            vehicle_id = self._find_key_from_value(self.ev_dict, i)
            edge_id = traci.vehicle.getRoute(vehicle_id)[0]

            # stop at the ending vertex of vehicle's starting edge
            # notice here each vehicle must finish traveling along it starting edge
            # there is no way to reassign it.
            self.routes.append(tuple([edge_id]))

            traci.vehicle.setStop(
                vehID=vehicle_id,
                edgeID=edge_id,
                pos=self.edge_length_dict[edge_id],
                laneIndex=0,
                duration=189999999999,
                flags=0,
                startPos=0,
            )

            print("Step stop for vehicle: ", vehicle_id)

    def _park_vehicle_to_assigned_starting_pos(self):
        print("Parking all car to their destinations of the starting edges...")
        while False in self.stop_statuses:
            traci.simulationStep()
            self._update_stop_status()
        print("All vehicles ready for model routing.")

    def _update_route_with_stop(self):
        """
        Update the route for each vehicle with the edge from its current stopped vertex to the next assigned vertex,
        and set the next stop to that 'to' vertex.
        Skip the vehicles that are still traveling along the edge, i.e., ev_stop = False.
        """
        eligible_vehicle = traci.vehicle.getIDList()
        for i in range(self.n_vehicle):
            if self.stop_statuses[i]:

                # handle the case when the destination of last demand is the start of current demand
                # i.e., the vehicle need to perform drop and pickup at the same location for two different demands correspondingly
                # so the location won't change, stop remains the same, no re-routing needed.
                if self.travel_info[i][0] == self.travel_info[i][1]:
                    continue

                self.stop_statuses[i] = False
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                if vehicle_id not in eligible_vehicle:
                    continue

                # all demands satisfied or being responding, remove all idle car to unblock others at junction
                if self.travel_info[i][1] == IDLE_LOCATION:
                    traci.vehicle.remove(vehicle_id)
                    print("Vehicle ", vehicle_id, " becomes idle, remove from network.")

                else:
                    traci.vehicle.resume(vehicle_id)

                    via_edge = self.travel_info[i]
                    edge_id = self._find_key_from_value(
                        self.edge_dict, self._find_edge_index(via_edge)
                    )

                    if "split" not in edge_id:
                        actual_edge_id = edge_id
                    else:
                        actual_edge_id = edge_id[7:]
                    self.routes[i] += tuple([actual_edge_id])

                    print(
                        "Vehicle ",
                        vehicle_id,
                        " has arrived stopped location, reassign new routes: ",
                        self.routes[i][-2],
                        self.routes[i][-1],
                    )
                    if (
                        self.routes[i][-1] != self.routes[i][-2]
                    ):  # handle the case for stopping at CS and then resume
                        traci.vehicle.setRoute(
                            vehID=vehicle_id, edgeList=self.routes[i][-2:]
                        )
                    else:
                        edge_id = actual_edge_id
                    traci.vehicle.setStop(
                        vehID=vehicle_id,
                        edgeID=actual_edge_id,
                        pos=self.edge_length_dict[edge_id],
                        laneIndex=0,
                        duration=189999999999,
                        flags=0,
                        startPos=0,
                    )

    def _update_stop_status(self):
        eligible_vehicle = traci.vehicle.getIDList()

        for i in range(self.n_vehicle):
            vehicle_id = self._find_key_from_value(self.ev_dict, i)
            if vehicle_id not in eligible_vehicle:
                continue
            if (
                traci.vehicle.getStopState(vehicle_id) == STOPPED_STATUS
            ):  # arrived the assigned vertex, can be assigned to the next
                self.stop_statuses[i] = True

    def _find_key_from_value(self, dict, value):
        return list(dict.keys())[list(dict.values()).index(value)]

    def _find_edge_index(self, via_edge):
        for i in range(len(self.edges)):
            if via_edge[0] == self.edges[i].start and via_edge[1] == self.edges[i].end:
                return i
