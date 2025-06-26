# environment.py

import numpy as np
from collections import deque
from config import (GRID_SIZE, INITIAL_MOBILE_VICTIMS,
                    INITIAL_DRIFTING_VICTIMS, VICTIM_MOVE_PROBABILITY, VICTIM_DRIFT_VECTOR,
                    NUM_RIVERS, RIVER_WIDTH)

class Victim:
    def __init__(self, id, pos, motion_type='mobile'):
        self.id = id; self.pos = np.array(pos, dtype=float)
        self.motion_type = motion_type
        self.history = [tuple(np.round(self.pos).astype(int))]
    def step(self, terrain_map):
        if self.motion_type == 'mobile':
            if np.random.random() < VICTIM_MOVE_PROBABILITY:
                moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]; np.random.shuffle(moves)
                for move in moves:
                    next_pos_candidate = self.pos + move
                    if self._is_valid(np.round(next_pos_candidate).astype(int), terrain_map):
                        self.pos = next_pos_candidate; break
        elif self.motion_type == 'drifting':
            move = np.array(VICTIM_DRIFT_VECTOR) + np.random.randn(2) * 0.1
            next_pos_candidate = self.pos + move
            if self._is_valid(np.round(next_pos_candidate).astype(int), terrain_map):
                self.pos = next_pos_candidate
        self.pos = np.clip(self.pos, 0, GRID_SIZE - 1)
        self.history.append(tuple(np.round(self.pos).astype(int)))
    def _is_valid(self, pos, terrain_map):
        H, W = terrain_map.shape
        if not (0 <= pos[0] < H and 0 <= pos[1] < W): return False
        if terrain_map[pos[0], pos[1]] == 1: return False
        return True
    @property
    def int_pos(self): return tuple(np.round(self.pos).astype(int))


class Environment:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.terrain_map = self._generate_terrain_with_rivers()
        self.reachable = self._mark_reachable()
        mask_unreachable = (~self.reachable) & (self.terrain_map == 0)
        self.terrain_map[mask_unreachable] = 1
        self.victims = self._generate_initial_victims()
        self.victim_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.update_victim_map()
        self.next_victim_id = len(self.victims)

    def update_victims(self):
        for victim in self.victims: victim.step(self.terrain_map)
        self.update_victim_map()

    def update_victim_map(self):
        self.victim_map.fill(False)
        for victim in self.victims:
            r, c = victim.int_pos
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size: self.victim_map[r, c] = True

    def spawn_new_victim(self, explored_map):
        unexplored_free_cells = np.argwhere(self.reachable & (self.terrain_map == 0) & ~explored_map)
        if len(unexplored_free_cells) < 10: return False
        spawn_pos = unexplored_free_cells[np.random.randint(len(unexplored_free_cells))]
        motion_type = 'mobile' if np.random.rand() > 0.5 else 'drifting'
        new_victim = Victim(id=f"V_{self.next_victim_id}", pos=spawn_pos, motion_type=motion_type)
        self.victims.append(new_victim); self.next_victim_id += 1
        print(f"    -> New '{motion_type}' victim spawned at {tuple(int(x) for x in spawn_pos)}")
        return True

    def _generate_terrain_with_rivers(self):
        """Generates a base terrain and then carves random rivers with arbitrary slopes."""
        terrain = (np.random.random((self.grid_size, self.grid_size)) > 0.98).astype(np.int8)

        for _ in range(NUM_RIVERS):
            # Select two random points on different edges for an arbitrary slope
            p1_edge = np.random.randint(4)
            p2_edge = (p1_edge + np.random.randint(1, 4)) % 4
            
            def get_point_on_edge(edge):
                if edge == 0: return np.array([0, np.random.randint(self.grid_size)]) # Top
                if edge == 1: return np.array([self.grid_size - 1, np.random.randint(self.grid_size)]) # Bottom
                if edge == 2: return np.array([np.random.randint(self.grid_size), 0]) # Left
                return np.array([np.random.randint(self.grid_size), self.grid_size - 1]) # Right
            
            p1 = get_point_on_edge(p1_edge).astype(float)
            p2 = get_point_on_edge(p2_edge).astype(float)

            # Use DDA line algorithm to draw the river path
            diff = p2 - p1
            steps = int(np.linalg.norm(diff)) * 2
            if steps == 0: continue
            
            increment = diff / steps
            direction_norm = increment / np.linalg.norm(increment)
            perp_norm = np.array([-direction_norm[1], direction_norm[0]])

            current_pos = p1.copy()
            for _ in range(steps):
                # Add meandering/wobble
                wobble = perp_norm * (np.random.random() - 0.5) * 4
                
                # Carve out the river width at the current position
                for w in range(-RIVER_WIDTH // 2, RIVER_WIDTH // 2 + 1):
                    offset = perp_norm * w
                    river_pos = current_pos + offset + wobble
                    r, c = np.round(river_pos).astype(int)
                    if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        terrain[r, c] = 1 # River is an obstacle
                
                current_pos += increment
                if not (0 <= current_pos[0] < self.grid_size and 0 <= current_pos[1] < self.grid_size):
                    break

        terrain[0,0] = 0 # Ensure start is always free
        return terrain

    def _mark_reachable(self):
        reachable = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        if self.terrain_map[0,0] == 1: return reachable
        queue = deque([(0, 0)]); reachable[0, 0] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if (0<=nx<self.grid_size and 0<=ny<self.grid_size and self.terrain_map[nx,ny]==0 and not reachable[nx,ny]):
                    reachable[nx,ny] = True; queue.append((nx,ny))
        return reachable

    def _generate_initial_victims(self):
        victims = []; free_cells = np.argwhere(self.reachable & (self.terrain_map == 0))
        total_victims = INITIAL_MOBILE_VICTIMS + INITIAL_DRIFTING_VICTIMS
        if len(free_cells) < total_victims: return []
        rng = np.random.default_rng(); chosen_indices = rng.choice(len(free_cells), size=total_victims, replace=False)
        victim_positions = free_cells[chosen_indices]
        for i in range(INITIAL_MOBILE_VICTIMS): victims.append(Victim(id=f"M_{i}", pos=victim_positions[i], motion_type='mobile'))
        for i in range(INITIAL_DRIFTING_VICTIMS):
            pos_idx = INITIAL_MOBILE_VICTIMS + i
            victims.append(Victim(id=f"D_{i}", pos=victim_positions[pos_idx], motion_type='drifting'))
        return victims