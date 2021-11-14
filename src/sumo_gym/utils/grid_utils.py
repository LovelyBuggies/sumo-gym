import sumo_gym.utils.network_utils as network_utils


def one_step_to_destination(vertices, edges, start_index, dest_index):
    visited = [False] * vertices

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
