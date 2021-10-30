# Environments MDP

## VRP

For simplicity, we assume vertices are all-connected.

- Observation: `locations`, the current locations of the vehicles; `loading`, the current loading of the vehicles.
  
- Action: `action_space` is determined by the current locations according to the adjacency list, as well as whether 
it is fully loaded.

- Transitions: if at the depot, unload; else choose one  avaliable and profitable vertex to go.

- Rewards: with `5 * loading` rewards and manhattan distance discount.


