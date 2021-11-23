# Environments MDP

## VRP

For simplicity, we assume vertices are all-connected.

- Observation: `locations`, the current locations of the vehicles; `loading`, the current loading of the vehicles.
  
- Action: `action_space` is determined by the current locations according to the adjacency list, as well as whether 
it is fully loaded.

- Transitions: if at the depot, unload; else choose one  available and profitable vertex to go.

- Rewards: with `5 * loading` rewards and manhattan distance discount.

## FMP 

MDP for FMP is defined only in SUMO-gym, w/o any strategies and policies (the order of responding demands).

- Observation:
  - `locations` the current locations of the vehicles;
  - `is_loading` whether the vehicles are loading passengers, what are the demand index to respond;
  - `is_charing` what are the charging station the vehicles are going to.
  - `batteries` what are the battery level of vehicles.

- Action: `action_space` is determined by the current state, returns a `GridAction` that contains information to update `is_loading`, `is_charging`, `locations`.

- Transition: for each vehicle
  - if is loading a passenger, change the `is_loading` according to whether arrive the destination;
  - if is responding a demand (but not loading), change the `is_loading` according to whether pick up the passenger;
  - if is charging, change `is_charging` according to whether has finished;
  - else, choose a demand to response, change the `is_loading` or go to charge randomly.
 
- Rewards: for each vehicle
  - reduce rewards for battery costs
  - add rewards for delivery, where hot spots have more rewards