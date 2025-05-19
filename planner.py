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

    # Check cache
    if path_cache is not None and (start, goal) in path_cache:
        path = path_cache[(start, goal)]
        # Validate: if any cell on cached path has become obstacle, invalidate
        for (i, j) in path:
            if terrain_map[i, j] not in (0, -1):
                path_cache.pop((start, goal), None)
                return astar(start, goal, terrain_map, path_cache)
        return path

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
            rev = []
            node = current
            while node != start:
                rev.append(node)
                node = came_from[node]
            rev.append(start)
            path = rev[::-1]
            if path_cache is not None:
                path_cache[(start, goal)] = path
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nei = (current[0] + dx, current[1] + dy)
            if (0 <= nei[0] < rows and 0 <= nei[1] < cols 
                and terrain_map[nei] in (0, -1)):
                tentative = g_score[current] + 1
                if tentative < g_score.get(nei, float('inf')):
                    g_score[nei] = tentative
                    came_from[nei] = current
                    f = tentative + heuristic(nei, goal)
                    heapq.heappush(open_heap, (f, nei))
    return []
