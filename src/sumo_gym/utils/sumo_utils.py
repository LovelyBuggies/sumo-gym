import traci
import sumolib
import os
import sys
from typing import Type, Tuple, Any, List


class SumoInteractionSingleton(object):
    #TODO add type hints once I figure out what type the vehicle/vehicle location is post-mapping
    def __init__(self, sumo_config: str, vehicles = None, locations = None):
        self.first_step = True
        self.sumo_init(sumo_config, vehicles, locations)

    #TODO add type hints and fmp->xml
    def sumo_init(self, sumo_config: str = None, vehicles = None, locations = None):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
        sumoCmd = [sumoBinary, "-c", sumo_config]
        traci.start(sumoCmd)

        # add any uninitialized vehicles into sumo
        if vehicles is not None and locations is not None:
            self.addVehicle(vehicles, locations)

    #TODO add type hints and integrate fmp -> xml mapping
    def addVehicle(self, vehicles=None, locations=None):
        if vehicles is not None and locations is not None:
            for vehicle, location in zip(vehicles, locations):
                traci.route.add(str(vehicle), [location])
                traci.vehicle.add(vehID=str(vehicle), routeID=str(vehicle))
                traci.vehicle.changeTarget(vehID=str(vehicle), edgeID=location)
                traci.vehicle.setStop(vehID=str(vehicle), edgeID=location, duration=float('inf'))

    #TODO type hints
    def removeVehicle(self, vehicles=None):
        if vehicles is not None:
            for vehicle in vehicles:
                traci.vehicle.remove(vehicle)

    #TODO integrate fmp -> xml mapping once I can get it
    def updateStop(self, vehicles=None, locations=None):
        if vehicles is not None and locations is not None:
            for vehicle, stop in zip(vehicles, locations):
                traci.vehicle.changeTarget(vehID=str(vehicle), edgeID=stop)

                if traci.vehicle.isStopped(str(vehicle)):
                    traci.vehicle.replaceStop(str(vehicle), 0, stop, duration=float('inf'))
                    #traci.vehicle.setStop(str(vehicle), )
                else:
                    traci.vehicle.setStop(vehID=str(vehicle), edgeID=stop, duration=float('inf'))

    #TODO add adjustment for multiple travel times after dynamic edge distance is implemented
    def simStep(self):
        if self.first_step:
            traci.simulationStep()
            self.first_step = False
        notStopped = True
        while notStopped:
            notStopped = False
            for vehicle in traci.vehicle.getIDList():
                if not traci.vehicle.isStopped(vehicle):
                    notStopped = True
            if notStopped:
                traci.simulationStep()

    def endSim(self):
        traci.close(False)
