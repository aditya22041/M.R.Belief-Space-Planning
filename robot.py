# robot.py

import numpy as np
import random
from belief import BeliefModule
from planner import astar
from allocator import compute_assignment
from config import (
    GRID_SIZE, SENSE_RANGE, ALLOC_INTERVAL, 
    TRUE_OBSTACLE_RATE, FALSE_OBSTACLE_RATE, ROBOT_SPEED,TRUE_POSITIVE_RATE,FALSE_POSITIVE_RATE
)

class Robot:
    """
    Each Robot maintains:
      - pos: current integer (i,j)
      - terrain_map: (H,W) with -1=unknown, 0=free, 1=obstacle
      - explored_map: bool mask of sensed cells
      - victim_map: bool mask of detected victims
      - confidence_map: BeliefModule over (free, obstacle, victim)
      - frontiers: set of unexplored neighbors
      - assignment_map: Voronoi partition (which robot “owns” each cell)
      - path: planned path to `target`
      - new_data_count / comm_count / step_count
      - reported_victims: grid of which victims have been broadcast
      - speed: number of cells moved per step
      - incoming_messages: list of dicts from CommunicationManager
    """

    def __init__(self, id, pos, grid_shape):
        self.id = id
        self.pos = list(pos)
        self.grid_shape = grid_shape
        self.speed = ROBOT_SPEED
        self.sense_range = SENSE_RANGE

        self.terrain_map = np.full(grid_shape, -1, dtype=np.int8)
        self.explored_map = np.zeros(grid_shape, dtype=bool)
        self.victim_map = np.zeros(grid_shape, dtype=bool)
        self.confidence_map = BeliefModule(grid_shape)

        self.path = []
        self.target = None
        self.confirmation_target = None

        self.new_data_count = 0
        self.comm_count = 0
        self.step_count = 0

        self.frontiers = set()
        self.assignment_map = None
        self.path_cache = {}

        # For frontier‐density scores
        self.unexplored_density = np.ones(grid_shape, dtype=np.float32)

        # Track which victims have been reported already
        self.reported_victims = np.zeros(grid_shape, dtype=bool)

        # Inbox for CommunicationManager
        self.incoming_messages = []

    def update_density(self, changed_cells=None):
        """
        Update the “unexplored‐density” for any cell in a 7×7 window around each changed cell,
        or recompute the entire grid if changed_cells is None.
        """
        H, W = self.grid_shape
        if changed_cells is None:
            for i in range(H):
                for j in range(W):
                    if self.explored_map[i, j]:
                        self.unexplored_density[i, j] = 0.0
                    else:
                        total = 0
                        count_unexpl = 0
                        for di in range(-3, 4):
                            for dj in range(-3, 4):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < H and 0 <= nj < W:
                                    total += 1
                                    if not self.explored_map[ni, nj]:
                                        count_unexpl += 1
                        self.unexplored_density[i, j] = (count_unexpl / total) if total > 0 else 0.0
        else:
            for (ci, cj) in changed_cells:
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        i, j = ci + di, cj + dj
                        if 0 <= i < H and 0 <= j < W:
                            if self.explored_map[i, j]:
                                self.unexplored_density[i, j] = 0.0
                            else:
                                total = 0
                                count_unexpl = 0
                                for di2 in range(-3, 4):
                                    for dj2 in range(-3, 4):
                                        ni, nj = i + di2, j + dj2
                                        if 0 <= ni < H and 0 <= nj < W:
                                            total += 1
                                            if not self.explored_map[ni, nj]:
                                                count_unexpl += 1
                                self.unexplored_density[i, j] = (count_unexpl / total) if total > 0 else 0.0

    def sense(self, true_terrain, true_victims):
        """
        Sense all cells within Manhattan radius = SENSE_RANGE.  Do noisy terrain + victim detection,
        then Bayesian‐update the belief.  Finally, update frontier set + local density + new_data_count.
        """
        x, y = self.pos
        H, W = self.grid_shape

        r_lo = max(0, x - self.sense_range)
        r_hi = min(H, x + self.sense_range + 1)
        c_lo = max(0, y - self.sense_range)
        c_hi = min(W, y + self.sense_range + 1)
        window_r, window_c = np.meshgrid(np.arange(r_lo, r_hi), np.arange(c_lo, c_hi), indexing='ij')
        window_coords = np.column_stack((window_r.ravel(), window_c.ravel()))

        mask_new = ~self.explored_map[window_coords[:,0], window_coords[:,1]]
        sense_cells = window_coords[mask_new]
        if sense_cells.size == 0:
            return

        changed = [tuple(x) for x in sense_cells.tolist()]

        # 1) Noisy terrain sensing
        true_vals = true_terrain[sense_cells[:,0], sense_cells[:,1]]
        terrain_detected = np.zeros(len(sense_cells), dtype=np.int8)
        for idx, is_obst in enumerate(true_vals):
            p = np.random.random()
            if is_obst:
                terrain_detected[idx] = 1 if p < TRUE_OBSTACLE_RATE else 0
            else:
                terrain_detected[idx] = 1 if p < FALSE_OBSTACLE_RATE else 0

        prev_terrain = self.terrain_map[sense_cells[:,0], sense_cells[:,1]].copy()
        self.explored_map[sense_cells[:,0], sense_cells[:,1]] = True
        self.terrain_map[sense_cells[:,0], sense_cells[:,1]] = terrain_detected

        # If a new obstacle blocks the current path, clear it
        new_obst = (terrain_detected == 1) & (prev_terrain != 1)
        if np.any(new_obst):
            self.path = []
            self.path_cache.clear()

        # 2) Victim detection on cells that were sensed free
        tv = true_victims[sense_cells[:,0], sense_cells[:,1]]
        free_idx = np.where(terrain_detected == 0)[0]
        detected = np.zeros(len(sense_cells), dtype=bool)
        for idx in free_idx:
            if tv[idx]:
                if np.random.random() < TRUE_POSITIVE_RATE:
                    detected[idx] = True
            else:
                if np.random.random() < FALSE_POSITIVE_RATE:
                    detected[idx] = True
        self.victim_map[sense_cells[:,0], sense_cells[:,1]] = detected

        # Build observation codes: 1=obstacle, 2=victim, 0=free/no‐victim
        obs_codes = []
        for idx, terr in enumerate(terrain_detected):
            if terr == 1:
                obs_codes.append(1)
            else:
                obs_codes.append(2 if detected[idx] else 0)

        # Bayesian update of belief
        pos_tuples = [tuple(x) for x in sense_cells.tolist()]
        self.confidence_map.bayesian_update(pos_tuples, obs_codes)

        # Increase new_data_count
        self.new_data_count += len(sense_cells)

        # Update local unexplored density around changed cells
        self.update_density(changed_cells=changed)

        # Update frontiers: any neighbor of newly explored cell that remains unexplored
        for (i, j) in changed:
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not self.explored_map[ni, nj]:
                    self.frontiers.add((ni, nj))
            self.frontiers.discard((i, j))

    def select_target(self, step, robots):
        """
        Decide next target cell:
          1) Recompute the assignment_map (Voronoi) every ALLOC_INTERVAL steps.
          2) If a confirmation_target exists, go there.
          3) Otherwise, pick among frontiers assigned to me by assignment_map, scoring by
             (density*10 + victim_prob*5 - 0.1*distance).  If none, fall back to a random
             unexplored cell in my region, then a random free cell, else stay put.
        """
        self.step_count = step
        H, W = self.grid_shape

        if step % ALLOC_INTERVAL == 0 or self.assignment_map is None:
            positions = [r.pos for r in robots]
            self.assignment_map = compute_assignment(positions)

        if self.confirmation_target is not None:
            self.target = self.confirmation_target
            return

        valid_frontiers = [
            (i, j) for (i, j) in self.frontiers
            if self.assignment_map[i, j] == self.id
        ]
        if valid_frontiers:
            best_score = -np.inf
            best_cell = None
            for (i, j) in valid_frontiers:
                dens = self.unexplored_density[i, j]
                vp = self.confidence_map.belief[i, j, 2]
                dist = abs(i - self.pos[0]) + abs(j - self.pos[1])
                score = dens * 10 + vp * 5 - 0.1 * dist
                if score > best_score:
                    best_score = score
                    best_cell = (i, j)
            self.target = best_cell
            return

        # If no valid frontier, pick random unexplored in my partition
        mask_unexpl = (~self.explored_map) & (self.assignment_map == self.id)
        unexpl_idxs = np.argwhere(mask_unexpl)
        if len(unexpl_idxs) > 0:
            idx = random.randint(0, len(unexpl_idxs) - 1)
            self.target = tuple(unexpl_idxs[idx])
            return

        # If no unexplored, pick a random free cell in my partition
        mask_free = (self.terrain_map == 0) & (self.assignment_map == self.id)
        free_idxs = np.argwhere(mask_free)
        if len(free_idxs) > 0:
            idx = random.randint(0, len(free_idxs) - 1)
            self.target = tuple(free_idxs[idx])
            return

        # Otherwise stay in place
        self.target = tuple(self.pos)

    def move(self, comm_manager):
        """
        (1) Broadcast newly high‐confidence victims (P(victim)>0.8 & not yet reported).
        (2) If at a confirmation_target, finalize that victim’s flag.
        (3) Otherwise, move up to `speed` steps along `path` toward `target` (replanning if needed).
        """

        # (1) Broadcast any new high‐confidence detections
        conf_mask = (
            (self.confidence_map.belief[:, :, 2] > 0.8) &
            self.victim_map &
            (~self.reported_victims)
        )
        new_vs = np.argwhere(conf_mask)
        for (xi, yi) in new_vs:
            comm_manager.broadcast_confirm_request(self.id, (xi, yi))
            self.reported_victims[xi, yi] = True

        # (2) If at confirmation_target, finalize that victim location
        if self.confirmation_target and tuple(self.pos) == self.confirmation_target:
            ci, cj = self.confirmation_target
            self.victim_map[ci, cj] = (self.confidence_map.belief[ci, cj, 2] > 0.8)
            self.confirmation_target = None

        # (3) If no target or at target, do nothing
        if self.target is None or tuple(self.pos) == self.target:
            return

        # Replan if path is empty or the target changed
        if (not self.path) or (self.path and self.path[-1] != self.target):
            self.path = astar(tuple(self.pos), self.target, self.terrain_map, self.path_cache)

        # Move up to `speed` steps along that path
        steps_to_take = min(self.speed, len(self.path))
        for _ in range(steps_to_take):
            if not self.path:
                break
            nxt = self.path.pop(0)
            if self.terrain_map[nxt[0], nxt[1]] in (0, -1):
                self.pos = [nxt[0], nxt[1]]
            else:
                # Path blocked by a new obstacle: clear it
                self.path = []
                break
