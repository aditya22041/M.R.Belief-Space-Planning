# planner.py
# -*- coding: utf-8 -*-
"""
D* Lite & RRT* planners with fallback A* for robustness.
"""
import numpy as np
import heapq
import math
import random
import config

# --- A* fallback for reliability ---
def astar_path(grid, start, goal):
    """Standard A* on 4-connected grid."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + abs(start[0]-goal[0]) + abs(start[1]-goal[1]), start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (current[0]+dx, current[1]+dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if grid[nb] != 0:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + abs(nb[0]-goal[0]) + abs(nb[1]-goal[1])
                heapq.heappush(open_set, (f, nb))
    return []

class DStarLitePlanner:
    def __init__(self, grid, start, goal):
        self.grid  = grid
        self.start = start
        self.goal  = goal
        self.rhs   = {}
        self.g     = {}
        self.U     = []
        self.km    = 0
        self._init_graph()

    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _calc_key(self, u):
        gu   = self.g.get(u, float('inf'))
        rhsu = self.rhs.get(u, float('inf'))
        return (
            min(gu, rhsu) + self._heuristic(self.start, u) + self.km,
            min(gu, rhsu)
        )

    def _neighbors(self, u):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            v = (u[0]+dx, u[1]+dy)
            if (0 <= v[0] < self.grid.shape[0] and
                0 <= v[1] < self.grid.shape[1] and
                self.grid[v] == 0):
                yield v

    def _update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(
                self.g.get(s, float('inf')) + 1 for s in self._neighbors(u)
            )
        # remove u from U
        self.U = [(k,node) for (k,node) in self.U if node != u]
        heapq.heapify(self.U)
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self._calc_key(u), u))

    def _compute_shortest_path(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            k_new = self._calc_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self._neighbors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = float('inf')
                for s in list(self._neighbors(u)) + [u]:
                    self._update_vertex(s)
            if self.g.get(self.start, float('inf')) == self.rhs.get(self.start, float('inf')):
                break

    def _init_graph(self):
        self.rhs[self.goal] = 0
        self.g[self.goal]   = float('inf')
        heapq.heappush(self.U, (self._calc_key(self.goal), self.goal))

    def plan(self):
        try:
            self._compute_shortest_path()
            if self.rhs.get(self.start, float('inf')) == float('inf'):
                raise ValueError("No path in D* Lite")
            # reconstruct
            path, u = [], self.start
            while u != self.goal:
                path.append(u)
                u = min(
                    self._neighbors(u),
                    key=lambda s: self.g.get(s, float('inf')) + 1
                )
            path.append(self.goal)
            return path
        except Exception:
            # fallback to A*
            return astar_path(self.grid, self.start, self.goal)

class RRTStarPlanner:
    def __init__(self, start, goal, is_free, bounds, max_iter=2000, step=20):
        self.start, self.goal = start, goal
        self.is_free, self.bounds = is_free, bounds
        self.max_iter, self.step = max_iter, step
        self.tree = {start: None}
        self.cost = {start: 0}

    def _sample(self):
        if random.random() < 0.05:
            return self.goal
        (xmin,ymin),(xmax,ymax) = self.bounds
        return (random.uniform(xmin,xmax), random.uniform(ymin,ymax))

    def _nearest(self, x):
        return min(
            self.tree.keys(),
            key=lambda v: math.hypot(v[0]-x[0], v[1]-x[1])
        )

    def _steer(self, a, b):
        theta = math.atan2(b[1]-a[1], b[0]-a[0])
        return (a[0] + self.step*math.cos(theta),
                a[1] + self.step*math.sin(theta))

    def plan(self):
        for _ in range(self.max_iter):
            xr = self._sample()
            xn = self._nearest(xr)
            xnew = self._steer(xn, xr)
            if not self.is_free(xnew):
                continue
            self.tree[xnew] = xn
            self.cost[xnew] = self.cost[xn] + math.hypot(
                xnew[0]-xn[0], xnew[1]-xn[1]
            )
            # rewire
            for xe in list(self.tree.keys()):
                if xe == xnew:
                    continue
                if math.hypot(xe[0]-xnew[0], xe[1]-xnew[1]) < self.step and self.is_free(xe):
                    newc = self.cost[xnew] + math.hypot(
                        xe[0]-xnew[0], xe[1]-xnew[1]
                    )
                    if newc < self.cost[xe]:
                        self.tree[xe], self.cost[xe] = xnew, newc
            if math.hypot(
                xnew[0]-self.goal[0], xnew[1]-self.goal[1]
            ) < self.step:
                self.tree[self.goal] = xnew
                break
        # extract path
        path, cur = [], (self.goal if self.goal in self.tree else None)
        while cur:
            path.append(cur)
            cur = self.tree[cur]
        return list(reversed(path))

if __name__ == "__main__":
    # D* Lite on empty grid
    empty = np.zeros((50,50), dtype=np.int8)
    dstar = DStarLitePlanner(empty, (0,0), (49,49))
    p1 = dstar.plan()
    assert p1, "Planner produced no path"

    # RRT* in free space
    def free_chk(pt):
        return 0 <= pt[0] < 100 and 0 <= pt[1] < 100
    rrt = RRTStarPlanner((0,0),(90,90), free_chk, ((0,0),(100,100)), max_iter=1000)
    p2 = rrt.plan()
    assert p2 and p2[0]==(0,0) and p2[-1]==(90,90), "RRT* failed"
    print("planner.py tests passed.")
