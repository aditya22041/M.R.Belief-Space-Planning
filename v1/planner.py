# planner.py

import heapq

def astar(start, goal, terrain_map, path_cache=None):
    """
    Standard A* on a 4‐connected grid. 
    Blocks: terrain_map[cell] == 1 (obstacle). Free if 0 or -1.
    Returns a list of (i,j) from start to goal (inclusive) or [] if no path.
    Uses a simple Manhattan‐distance heuristic.
    """

    if start == goal:
        return [start]

    # Check cache if available
    if path_cache is not None and (start, goal) in path_cache:
        path = path_cache[(start, goal)]
        # Validate cache: check if path is still valid
        for (i, j) in path:
            if terrain_map[i, j] == 1:
                path_cache.pop((start, goal))
                # Fallthrough to re-compute
                break
        else: # No break
            return list(path) # Return a copy

    rows, cols = terrain_map.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    g_score = {start: 0}
    f_score = heuristic(start, goal)
    heapq.heappush(open_heap, (f_score, start))
    came_from = {}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path = path[::-1]
            if path_cache is not None:
                path_cache[(start, goal)] = tuple(path) # Cache as tuple
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if terrain_map[neighbor[0], neighbor[1]] == 1: # Obstacle
                continue
            
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, neighbor))
    return [] # No path found