from typing import Type, Tuple, Any
import numpy as np
import numpy.typing as npt
from gym.spaces import Space


VertexType = list[Any]
VerticesType = npt.NDArray[VertexType] # [id (str), x-position (float), y-position (float)]
DemandType = npt.NDArray[float]
EdgeType = npt.NDArray[Tuple[Tuple[str], float]]
DeparturesType = npt.NDArray[int]
CapacityType = npt.NDArray[float]
AdjListType = npt.NDArray[npt.NDArray[int]]
LocationsType = npt.NDArray[int]
LoadingType = npt.NDArray[Tuple[float]]
SpaceType = Space
ActionsType = npt.NDArray[int]
RewardsType = npt.NDArray[float]

FMPElectricVehiclesType = npt.NDArray[Tuple[str, float]]     # vehicle index, chargeDelay
FMPChargingStationType = npt.NDArray[Tuple[str, float]]        # vertex index, indicator
FMPDemandsType = npt.NDArray[Tuple[int]]                # departure, destination