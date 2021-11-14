import sumo_gym.utils.network_utils as network_utils


def one_step_to_destination(vertices, edges, start_index, dest_index):
    visited = [False] * len(vertices)
    bfs_queue = [dest_index]
    visited[dest_index] = True

    while bfs_queue:
        curr = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == start_index:
                return curr
            elif not visited[v]:
                bfs_queue.append(v)
                visited[v] = False


def nearest_charging_station_with_distance(vertices, charging_stations, edges, start_index):
    charging_station_vertices = [charging_station[0] for charging_station in charging_stations]
    visited = [False] * len(vertices)

    bfs_queue = [[start_index, 0]]
    visited[start_index] = True

    while bfs_queue:
        curr, curr_depth = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v in charging_station_vertices:
                return charging_station_vertices.index(v), curr_depth + 1
            elif not visited[v]:
                bfs_queue.append([v, curr_depth+1])
                visited[v] = False


def dist_between(vertices, edges, start_index, dest_index):
    visited = [False] * len(vertices)

    bfs_queue = [[start_index, 0]]
    visited[start_index] = True

    while bfs_queue:
        curr, curr_depth = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == dest_index:
                return curr_depth + 1
            elif not visited[v]:
                bfs_queue.append([v, curr_depth+1])
                visited[v] = False


def get_hot_spot_weight(vertices, edges, demands, demand_start):
    total_demands = len(demands)

    adjacent_vertices = network_utils.get_adj_list(vertices, edges)[demand_start]
    adjacent_vertices.append(demand_start)
    local_demands = len([d for d in demands if d[0] in adjacent_vertices])

    return local_demands / total_demands * 100

