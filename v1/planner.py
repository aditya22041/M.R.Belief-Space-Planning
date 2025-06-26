# planner.py
import heapq
def astar(start, goal, terrain_map, path_cache=None):
    if start == goal: return [start]
    if path_cache is not None and (start, goal) in path_cache:
        path = path_cache[(start, goal)]
        if all(terrain_map[i, j] != 1 for i,j in path): return list(path)
        else: path_cache.pop((start, goal), None)
    rows, cols = terrain_map.shape
    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_heap = []; g_score = {start:0}; heapq.heappush(open_heap, (heuristic(start, goal), start)); came_from = {}
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []; node = current
            while node in came_from: path.append(node); node = came_from[node]
            path.append(start); path = path[::-1]
            if path_cache is not None: path_cache[(start, goal)] = tuple(path)
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if not (0<=neighbor[0]<rows and 0<=neighbor[1]<cols) or terrain_map[neighbor]==1: continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current; g_score[neighbor] = tentative_g_score
                heapq.heappush(open_heap, (tentative_g_score + heuristic(neighbor, goal), neighbor))
    return []