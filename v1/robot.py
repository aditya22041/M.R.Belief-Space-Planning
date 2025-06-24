# robot.py

import numpy as np
import random
from planner import astar
from config import (
    GRID_SIZE, SENSE_RANGE, ROBOT_SPEED,
    TRUE_OBSTACLE_RATE, FALSE_OBSTACLE_RATE, TRUE_POSITIVE_RATE, FALSE_POSITIVE_RATE,
    MAP_UPDATE_THRESHOLD, VICTIM_CONFIDENCE_THRESHOLD
)

class Robot:
    """
    Represents a single robot. Its sole purpose is to explore its assigned
    square regions and report findings to a central server.
    """
    def __init__(self, id, pos, grid_shape, server):
        self.id = id
        self.pos = list(pos)
        self.grid_shape = grid_shape
        self.server = server
        self.speed = ROBOT_SPEED
        self.sense_range = SENSE_RANGE

        # Local maps
        self.terrain_map = np.full(grid_shape, -1, dtype=np.int8)
        self.explored_map = np.zeros(grid_shape, dtype=bool)
        self.victim_confidence_map = np.zeros(grid_shape, dtype=float)

        self.path = []
        self.target = None
        
        self.newly_sensed_cells = []

        self.frontiers = set()
        self.assignment_map = None # Provided by the server
        self.path_cache = {}

    def sense(self, true_terrain, true_victims):
        """
        Sense the local environment for terrain and victims with noise.
        """
        x, y = self.pos
        H, W = self.grid_shape
        min_r, max_r = max(0, x - self.sense_range), min(H, x + self.sense_range + 1)
        min_c, max_c = max(0, y - self.sense_range), min(W, y + self.sense_range + 1)
        
        sensed_coords = []
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                if not self.explored_map[r, c]:
                    sensed_coords.append((r,c))

        if not sensed_coords:
            return

        self.newly_sensed_cells.extend(sensed_coords)
        coords_arr = np.array(sensed_coords)
        coords_idx = (coords_arr[:, 0], coords_arr[:, 1])
        self.explored_map[coords_idx] = True
        
        true_terrain_vals = true_terrain[coords_idx]
        for i, is_obstacle in enumerate(true_terrain_vals):
            (r, c) = sensed_coords[i]
            detected_obstacle = (is_obstacle and np.random.random() < TRUE_OBSTACLE_RATE) or \
                                (not is_obstacle and np.random.random() < FALSE_OBSTACLE_RATE)
            self.terrain_map[r, c] = 1 if detected_obstacle else 0
        
        true_victim_vals = true_victims[coords_idx]
        for i, is_victim in enumerate(true_victim_vals):
            (r,c) = sensed_coords[i]
            if self.terrain_map[r,c] == 0:
                detected_victim = (is_victim and np.random.random() < TRUE_POSITIVE_RATE) or \
                                  (not is_victim and np.random.random() < FALSE_POSITIVE_RATE)
                if detected_victim:
                    confidence = TRUE_POSITIVE_RATE if is_victim else FALSE_POSITIVE_RATE
                    self.victim_confidence_map[r, c] = max(self.victim_confidence_map[r, c], confidence)

        for r, c in sensed_coords:
            self.frontiers.discard((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not self.explored_map[nr, nc]:
                    self.frontiers.add((nr, nc))
    
    def select_target(self):
        """
        Selects a target based on the closest frontier in its assigned region.
        """
        if self.assignment_map is None:
            self.target = tuple(self.pos)
            return

        my_frontiers = [f for f in self.frontiers if self.assignment_map[f[0], f[1]] == self.id]
        
        if not my_frontiers:
            mask = (~self.explored_map) & (self.assignment_map == self.id)
            unexplored_indices = np.argwhere(mask)
            if unexplored_indices.size > 0:
                self.target = tuple(unexplored_indices[np.random.randint(len(unexplored_indices))])
            else:
                self.target = tuple(self.pos)
            return

        self.target = min(my_frontiers, key=lambda f: abs(f[0] - self.pos[0]) + abs(f[1] - self.pos[1]))

    def move(self):
        """
        Move towards the current target using A* planning.
        """
        if self.target is None or tuple(self.pos) == self.target:
            return
            
        if not self.path or self.path[-1] != self.target:
            self.path = astar(tuple(self.pos), self.target, self.terrain_map, self.path_cache)

        steps_taken = 0
        while self.path and steps_taken < self.speed:
            next_pos = self.path.pop(0)
            if self.terrain_map[next_pos[0], next_pos[1]] != 1:
                self.pos = list(next_pos)
                steps_taken += 1
            else: 
                self.path = [] 
                self.path_cache.clear()
                break

    def communicate(self):
        """
        Send map updates to the central server if the threshold is met.
        """
        if len(self.newly_sensed_cells) >= MAP_UPDATE_THRESHOLD:
            positions = self.newly_sensed_cells
            coords = np.array(positions).T
            terrain_vals = self.terrain_map[tuple(coords)].tolist()
            confidence_vals = self.victim_confidence_map[tuple(coords)].tolist()
            explored_vals = self.explored_map[tuple(coords)].tolist()
            
            msg = {
                'sender_id': self.id, 
                'type': 'map_update', 
                'positions': positions, 
                'terrain': terrain_vals, 
                'confidence': confidence_vals,
                'explored': explored_vals,
            }
            self.server.incoming_messages.append(msg)
            self.newly_sensed_cells = []
                
    def step(self, true_terrain, true_victims):
        """
        Execute one full step of the robot's logic.
        """
        self.select_target()
        self.move()
        self.sense(true_terrain, true_victims)
        self.communicate()