# environment.py

import numpy as np
from noise import pnoise2
from collections import deque
from config import GRID_SIZE, OBSTACLE_THRESHOLD, VICTIM_DENSITY

class Environment:
    """
    Generates a Perlin‐noise terrain, flood‐fills enclosed loops, computes reachability
    from (0,0), and randomly distributes victims on reachable free cells.
    """

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.terrain_map = self._generate_terrain()
        self._fill_closed_loops()
        self.reachable = self._mark_reachable()

        # Any free cell that is not reachable becomes an obstacle
        mask_unreachable = (~self.reachable) & (self.terrain_map == 0)
        self.terrain_map[mask_unreachable] = 1

        self.victim_map = self._generate_victims()

    def _generate_terrain(self):
        """
        Use Perlin noise to create smooth‐looking terrain in [0,1], then threshold at OBSTACLE_THRESHOLD.
        """
        world = np.zeros((self.grid_size, self.grid_size))
        scale = 15.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                world[i, j] = pnoise2(
                    i / scale, j / scale,
                    octaves=4,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=self.grid_size,
                    repeaty=self.grid_size,
                    base=0
                )
        world = (world - world.min()) / (world.max() - world.min())
        terrain = (world > OBSTACLE_THRESHOLD).astype(np.int8)
        terrain[0, 0] = 0  # Ensure the start is free
        return terrain

    def _fill_closed_loops(self):
        """
        Flood‐fill from any border free cell to mark all outside‐connected free cells.
        Any free cell not reached by that flood must be enclosed → fill with obstacle.
        """
        outside = np.ones((self.grid_size, self.grid_size), dtype=bool)
        queue = deque()

        # Enqueue all border free cells
        for i in range(self.grid_size):
            for j in [0, self.grid_size - 1]:
                if self.terrain_map[i, j] == 0 and outside[i, j]:
                    queue.append((i, j))
                    outside[i, j] = False
            for j2 in range(1, self.grid_size - 1):
                for i2 in [0, self.grid_size - 1]:
                    if self.terrain_map[i2, j2] == 0 and outside[i2, j2]:
                        queue.append((i2, j2))
                        outside[i2, j2] = False

        # Flood‐fill reachable free cells from border
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                        and self.terrain_map[nx, ny] == 0 
                        and outside[nx, ny]):
                    outside[nx, ny] = False
                    queue.append((nx, ny))

        # Any true in `outside` that is still free must be enclosed → obstacle
        enclosed_mask = outside & (self.terrain_map == 0)
        self.terrain_map[enclosed_mask] = 1

    def _mark_reachable(self):
        """
        Starting from (0,0), flood‐fill to mark all free cells reachable by 4‐connectivity.
        """
        reachable = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = deque([(0, 0)])
        reachable[0, 0] = True

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                        and self.terrain_map[nx, ny] == 0 
                        and not reachable[nx, ny]):
                    reachable[nx, ny] = True
                    queue.append((nx, ny))
        return reachable

    def _generate_victims(self):
        """
        Place ~VICTIM_DENSITY fraction of reachable free cells as victims (boolean mask).
        """
        victims = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        free_cells = np.argwhere(self.reachable & (self.terrain_map == 0))
        num_victims = max(1, int(VICTIM_DENSITY * len(free_cells)))
        rng = np.random.default_rng()
        chosen = rng.choice(len(free_cells), size=num_victims, replace=False)
        victims[free_cells[chosen, 0], free_cells[chosen, 1]] = True
        return victims
