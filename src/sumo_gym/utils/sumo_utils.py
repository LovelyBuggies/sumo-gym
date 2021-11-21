import os, sys
import traci
import traci.constants as tc

import sumo_gym.typing


SUMO_COMMAND = ["/opt/homebrew/opt/sumo/share/sumo/bin/sumo-gui", "-c", "assets/data/sumo_simulation.sumocfg"]


class SumoRender:
    def __init__(
        self,
        vertex_dict:    dict = None,
        edge_dict:  dict = None,
        ev_dict: dict = None,
        edges: sumo_gym.typing.EdgeType = None
        ):

        self.vertex_dict = vertex_dict
        self.edge_dict = edge_dict
        self.ev_dict = ev_dict
        self.edges = edges
        self.initialized = False

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        
        traci.start(SUMO_COMMAND)


    def render(self, prev_locations, actions):
        print("     ", prev_locations, actions)
        # traci.vehicle.setSpeed("vehicle_0", 23.45)

        #traci.vehicle.setStop(vehID="vehicle_1", edgeID="gneE2", pos=18.80, laneIndex=0, duration=189999999999, flags=0, startPos=0)
        
        if not self.initialized:
            self.initialized = True
            # change 'from', edge accordingly
            for i in range(len(actions)):
                
                vehicle_id = self._find_key_from_value(self.ev_dict, i)
                traci.vehicle.setSpeedMode(vehicle_id, 32) # all checks off

                via_edge = (prev_locations[i], actions[i].location)
                print("     via_edge: ", via_edge, vehicle_id)

                edge_id = self._find_key_from_value(self.edge_dict, self._find_edge_index(via_edge))
                print("     edge id: ", edge_id)
                
                original_routes = traci.vehicle.getRoute(vehID=vehicle_id)
                new_routes = (original_routes[0], edge_id)
                print("     new routes: ", new_routes)
                traci.vehicle.setRoute(vehID=vehicle_id, edgeList=new_routes)
                traci.vehicle.setStop(vehID=vehicle_id, edgeID=edge_id, pos=10.60, laneIndex=0, duration=189999999999, flags=0, startPos=0)
                #print("Stop state:", traci.vehicle.getStopState("vehicle_1"))
       
        for i in range(len(actions)):
            if prev_locations[i] != actions[i].location: 
                via_edge = (prev_locations[i], actions[i].location)
                edge_id = self._find_key_from_value(self.edge_dict, self._find_edge_index(via_edge))
                #traci.vehicle.setStop(vehID="vehicle_0", edgeID="gneE48", pos=22.60, laneIndex=0, duration=189999999999, flags=0, startPos=0)


        # TODO: append next edge to route, and change 'to' attr of trip
            
        
        traci.simulationStep()
        


    def close(self):
        traci.close()


    def _find_key_from_value(self, dict, value):
        return list(dict.keys())[list(dict.values()).index(value)]

    def _find_edge_index(self, via_edge):
        for i in range(len(self.edges)):
            if via_edge[0] == self.edges[i].start and via_edge[1] == self.edges[i].end:
                return i
