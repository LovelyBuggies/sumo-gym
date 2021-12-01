import os
import sys
import traci

import sumo_gym.typing


STOPPED_STATUS = 1


class SumoRender:
    def __init__(
        self,
        sumo_gui_path: str = None,
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

        traci.start([self.sumo_gui_path, "-c", "assets/data/sumo_simulation.sumocfg"])

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

        self._update_route_with_stop()
        traci.simulationStep()
        self._update_stop_status()

    def close(self):
        while not self.terminated:
            traci.simulationStep()

            self.terminated = True
            for i in range(self.n_vehicle):
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
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
            print("Step stop for vehicle: ", vehicle_id)

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

    def _update_route_with_stop(self):
        """
        Update the route for each vehicle with the edge from its current stopped vertex to the next assigned vertex,
        and set the next stop to that 'to' vertex.
        Skip the vehicles that are still traveling along the edge, i.e., ev_stop = False.
        """
        for i in range(self.n_vehicle):
            if self.stop_statuses[i]:

                # handle the case when the destination of last demand is the start of current demand
                # i.e., the vehicle need to perform drop and pickup at the same location for two different demands correspondingly
                # so the location won't change, stop remains the same, no re-routing needed.
                if self.travel_info[i][0] == self.travel_info[i][1]:
                    continue

                self.stop_statuses[i] = False
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                traci.vehicle.resume(vehicle_id)

                via_edge = self.travel_info[i]
                edge_id = self._find_key_from_value(
                    self.edge_dict, self._find_edge_index(via_edge)
                )
                self.routes[i] += tuple([edge_id])

                print(
                    "Vehicle ",
                    vehicle_id,
                    " has arrived stopped location, reassign new routes: ",
                    self.routes[i],
                )

                traci.vehicle.setRoute(vehID=vehicle_id, edgeList=self.routes[i][-2:])
                traci.vehicle.setStop(
                    vehID=vehicle_id,
                    edgeID=edge_id,
                    pos=self.edge_length_dict[edge_id],
                    laneIndex=0,
                    duration=189999999999,
                    flags=0,
                    startPos=0,
                )

    def _update_stop_status(self):
        for i in range(self.n_vehicle):
            vehicle_id = self._find_key_from_value(self.ev_dict, i)
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
