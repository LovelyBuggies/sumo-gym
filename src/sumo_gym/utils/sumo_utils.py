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
        vertices: sumo_gym.typing.VerticesType = None,
        vertex_dict: dict = None,
        n_vehicle: int = 1,
    ):
        self.sumo_gui_path = sumo_gui_path
        self.edge_dict = edge_dict
        self.edge_length_dict = edge_length_dict
        self.ev_dict = ev_dict
        self.edges = edges
        self.vertices = vertices
        self.vertex_dict = vertex_dict
        self.initialized = False
        self.terminated = False
        self.need_action = [False] * n_vehicle
        self.stopped = [None] * n_vehicle
        self.n_vehicle = n_vehicle
        self.routes = []
        self.last_edge = {
            i: None for i in range(n_vehicle)
        }  # only used for setting stop related usage

        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        traci.start([self.sumo_gui_path, "-c", sumo_config_path])

    def retrieve_need_action_status(self):
        return self.need_action

    def retrieve_stop_status(self):
        return self.stopped

    def update_travel_vertex_info_for_vehicle(self, vehicle_travel_info_list):
        self.travel_info = vehicle_travel_info_list

    def render(self):
        if not self.initialized:
            print("Initialize the first starting edge for vehicle...")
            self._initialize_route()
            self._park_vehicle_to_assigned_starting_pos()
            self.initialized = True
        else:
            self._update_route_with_stop()
            traci.simulationStep()
            self._update_need_action_status()

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
            print("====== ", traci.vehicle.getIDList(), traci.vehicle.getIDCount())
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
            self.last_edge[i] = edge_id

    def _park_vehicle_to_assigned_starting_pos(self):
        print("Parking all car to their destinations of the starting edges...")
        while False in self.need_action:
            traci.simulationStep()
            self._update_need_action_status()
        print("All vehicles ready for model routing.")

    def _update_route_with_stop(self):
        """
        Update the route for each vehicle with the edge from its current stopped vertex to the next assigned vertex,
        and set the next stop to that 'to' vertex.
        Skip the vehicles that are still traveling along the edge, i.e., ev_stop = False.
        """
        eligible_vehicle = traci.vehicle.getIDList()
        for i in range(self.n_vehicle):
            vehicle_id = self._find_key_from_value(self.ev_dict, i)

            if self.need_action[i]:
                if self.travel_info[i] is None:  # location not changed
                    traci.vehicle.setStop(
                        vehID=vehicle_id,
                        edgeID=self.routes[i][-1],
                        pos=self.edge_length_dict[self.last_edge[i]],
                        laneIndex=0,
                        duration=189999999999,
                        flags=0,
                        startPos=0,
                    )
                    continue

                if (
                    self.travel_info[i][1] == IDLE_LOCATION
                ):  # vehicle done, let it move to dest and disappear in network
                    print(
                        "Vehicle ",
                        vehicle_id,
                        " becomes idle, will disappear after finishing the trip.",
                    )
                else:
                    if len(traci.vehicle.getStops(vehID=vehicle_id, limit=1)) == 1:
                        traci.vehicle.replaceStop(
                            vehID=vehicle_id, nextStopIndex=0, edgeID=""
                        )

                    self.need_action[i] = False
                    if vehicle_id not in eligible_vehicle:
                        continue

                    via_edge = self.travel_info[i]
                    edge_id = self._find_key_from_value(
                        self.edge_dict, self._find_edge_index(via_edge)
                    )

                    if "split" not in edge_id:
                        actual_edge_id = edge_id
                    else:  # next action is charging station, need stop
                        actual_edge_id = edge_id[7:]

                    if (
                        self.routes[i][-1] != actual_edge_id
                    ):  # handle the case for stopping at CS and then resume
                        self.routes[i] += tuple([actual_edge_id])
                        self.last_edge[i] = edge_id

                        print(
                            "Vehicle ",
                            vehicle_id,
                            " : set next stop to: ",
                            self.routes[i][-1],
                        )

                        route_ind = traci.vehicle.getRouteIndex(vehicle_id)
                        traci.vehicle.setRoute(
                            vehID=vehicle_id,
                            edgeList=self.routes[i][route_ind - len(self.routes[i]) :],
                        )

                        traci.vehicle.setStop(
                            vehID=vehicle_id,
                            edgeID=actual_edge_id,
                            pos=self.edge_length_dict[edge_id],
                            laneIndex=0,
                            duration=189999999999,
                            flags=0,
                            startPos=0,
                        )
                    elif (
                        "split" in self.last_edge[i]
                        and self.last_edge[i][7:] == actual_edge_id
                    ):
                        self.last_edge[i] = actual_edge_id
                        traci.vehicle.setStop(
                            vehID=vehicle_id,
                            edgeID=actual_edge_id,
                            pos=self.edge_length_dict[actual_edge_id],
                            laneIndex=0,
                            duration=189999999999,
                            flags=0,
                            startPos=0,
                        )

    def _update_need_action_status(self):
        eligible_vehicle = traci.vehicle.getIDList()

        for i in range(self.n_vehicle):
            vehicle_id = self._find_key_from_value(self.ev_dict, i)
            if vehicle_id not in eligible_vehicle:
                continue
            elif (
                traci.vehicle.getDrivingDistance(
                    vehicle_id,
                    self.routes[i][-1],
                    self.edge_length_dict[self.last_edge[i]],
                )
                <= 20
            ):  # arriving the assigned vertex, can take the next action
                self.need_action[i] = True
                if traci.vehicle.getStopState(vehicle_id):
                    self.stopped[i] = traci.vehicle.getLanePosition(vehicle_id)
                else:
                    self.stopped[i] = None
            else:
                print("============> ", vehicle_id, " traveling =====================> ", traci.vehicle.getLaneID(vehicle_id), traci.vehicle.getSpeedMode(vehicle_id), traci.vehicle.getStopState(vehicle_id), self.last_edge[i])
                if traci.vehicle.getStopState(vehicle_id):
                    traci.vehicle.resume(vehicle_id)
                    print("============> Trying to resume vehicle: ", vehicle_id)
                self.need_action[i] = False

    def _find_key_from_value(self, dict, value):
        return list(dict.keys())[list(dict.values()).index(value)]

    def _find_edge_index(self, via_edge):
        for i in range(len(self.edges)):
            if via_edge[0] == self.edges[i].start and via_edge[1] == self.edges[i].end:
                return i
