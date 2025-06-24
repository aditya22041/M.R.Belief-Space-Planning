# central_server.py

import numpy as np
from region_assigner import assign_square_regions

class CentralServer:
    """
    Manages global state and assigns exploration regions to robots.
    This version uses an entropy-based square region assignment.
    """
    def __init__(self, grid_shape, robots):
        self.grid_shape = grid_shape
        self.robots = {r.id: r for r in robots}

        # Global maps
        self.global_terrain_map = np.full(grid_shape, -1, dtype=np.int8)
        self.global_explored_map = np.zeros(grid_shape, dtype=bool)
        self.global_confidence_map = np.zeros(grid_shape, dtype=float) + 0.5 # Start with max uncertainty
        
        # Communication
        self.incoming_messages = []
        self.comm_count = 0

    def update_assignments(self):
        """
        Compute and distribute new square-based assignments to all robots.
        """
        assignment_map = assign_square_regions(
            list(self.robots.values()),
            self.global_confidence_map,
            self.global_explored_map,
            self.grid_shape
        )
        for r in self.robots.values():
            r.assignment_map = assignment_map

    def process_messages(self):
        """
        Process all messages sent by robots in the current step.
        """
        while self.incoming_messages:
            msg = self.incoming_messages.pop(0)
            self.comm_count += 1
            
            if msg['type'] == 'map_update':
                self._merge_map_update(msg)

    def _merge_map_update(self, msg):
        """
        Merge a robot's local map data into the global map using weighted average for confidence.
        """
        positions = msg['positions']
        terrain_vals = msg['terrain']
        confidence_vals = msg['confidence']
        explored_vals = msg['explored']

        pos_array = np.array(positions)
        pos_idx = (pos_array[:,0], pos_array[:,1])

        # Update explored map
        self.global_explored_map[pos_idx] = np.logical_or(self.global_explored_map[pos_idx], explored_vals)
        
        # Update terrain
        self.global_terrain_map[pos_idx] = np.where(self.global_terrain_map[pos_idx] == -1, terrain_vals, self.global_terrain_map[pos_idx])
        
        # Update confidence with weighted merge
        b_old = self.global_confidence_map[pos_idx]
        b_new = np.array(confidence_vals)
        w_old = np.abs(b_old - 0.5)
        w_new = np.abs(b_new - 0.5)
        
        denominator = w_old + w_new
        # Avoid division by zero
        safe_denom = np.where(denominator == 0, 1, denominator)
        
        merged_belief = (w_old * b_old + w_new * b_new) / safe_denom
        
        # Where denominator was 0, just average
        merged_belief = np.where(denominator == 0, (b_old + b_new) / 2, merged_belief)
        
        self.global_confidence_map[pos_idx] = merged_belief


    def step(self, step_num):
        """
        A full step of the Central Server's logic.
        """
        self.process_messages()
        if step_num > 0 and step_num % 40 == 0:
             self.update_assignments()