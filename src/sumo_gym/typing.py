from typing import Type, Tuple, Any, List
import numpy as np
import numpy.typing as npt
from gym.spaces import Space
import sumo_gym


VertexType = Tuple[float]
VerticesType = npt.NDArray[VertexType]
DemandType = npt.NDArray[float]
EdgeType = npt.NDArray[Tuple[int]]
DeparturesType = npt.NDArray[int]
CapacityType = npt.NDArray[float]
AdjListType = npt.NDArray[npt.NDArray[int]]
LocationsType = npt.NDArray[int]
LoadingType = npt.NDArray[Tuple[float]]
SpaceType = Space
ActionsType = npt.NDArray[int]
RewardsType = npt.NDArray[float]
