import numpy as np
from region_assigner import assign_square_regions
from config import (ADAPTIVE_DECAY_ENABLED, INITIAL_BELIEF_HALF_LIFE, MIN_HALF_LIFE, MAX_HALF_LIFE,
                    BELIEF_DECAY_INTERVAL, REASSIGN_INTERVAL, SCENT_MAP_DECAY)

class CentralServer:
    def __init__(self, grid_shape, robots):
        self.grid_shape = grid_shape
        self.robots = {r.id: r for r in robots}
        self.global_terrain_map = np.full(grid_shape, -1, dtype=np.int8)
        self.global_explored_map = np.zeros(grid_shape, dtype=bool)
        
        # Two-Layer Belief System
        self.scent_map = np.zeros(grid_shape, dtype=float)  # Fast-decaying, for immediate reaction
        self.belief_map = np.zeros(grid_shape, dtype=float)  # Slow, long-term memory (victim probability)
        
        self.incoming_messages = []
        self.comm_count = 0
        self.adaptive_half_life = INITIAL_BELIEF_HALF_LIFE
        self.last_hotspot_count = 0

    def update_assignments(self):
        assignment_map = assign_square_regions(
            list(self.robots.values()),
            self.belief_map,
            self.global_explored_map,
            self.grid_shape
        )
        for r in self.robots.values():
            r.assignment_map = assignment_map

    def _merge_map_update(self, msg):
        # Unpack positions
        positions = np.array(msg['positions'])
        pos_idx = (positions[:, 0], positions[:, 1])

        # Merge explored and terrain
        self.global_explored_map[pos_idx] = np.logical_or(
            self.global_explored_map[pos_idx],
            msg['explored']
        )
        self.global_terrain_map[pos_idx] = np.where(
            self.global_terrain_map[pos_idx] == -1,
            msg['terrain'],
            self.global_terrain_map[pos_idx]
        )
        
        # Update fast-layer scent map
        self.scent_map[pos_idx] = np.maximum(
            self.scent_map[pos_idx],
            msg['scent']
        )

        # Update slow-layer belief map (victim probability fusion)
        new_b = msg['belief']
        if new_b.size:
            old_b = self.belief_map[pos_idx]
            # Weight new observations more when they are extreme (near 0 or 1)
            # shock factor ranges from 0 at 0.5 to 1 at 0 or 1
            shock = 2 * np.abs(new_b - 0.5)
            # Define weights: base weight 1 for old, base 1 plus shock for new
            w_old = 1.0
            w_new = 1.0 + shock
            # Compute weighted average
            merged = (w_old * old_b + w_new * new_b) / (w_old + w_new)
            # Clip and assign
            self.belief_map[pos_idx] = np.clip(merged, 0.0, 1.0)

    def adapt_belief_dynamics(self):
        if not ADAPTIVE_DECAY_ENABLED:
            return
        current_hotspot_count = np.sum(self.belief_map > 0.1)
        if current_hotspot_count < self.last_hotspot_count * 0.9:
            self.adaptive_half_life *= 1.05
        elif current_hotspot_count > self.last_hotspot_count * 1.2:
            self.adaptive_half_life *= 0.95
        self.adaptive_half_life = np.clip(
            self.adaptive_half_life,
            MIN_HALF_LIFE,
            MAX_HALF_LIFE
        )
        self.last_hotspot_count = current_hotspot_count

    def decay_maps(self):
        # Decay fast-layer scent map
        self.scent_map *= SCENT_MAP_DECAY
        self.scent_map[self.scent_map < 1e-3] = 0

        # Decay slow-layer belief map periodically
        if self.step_num > 0 and self.step_num % BELIEF_DECAY_INTERVAL == 0:
            self.adapt_belief_dynamics()
            decay_factor = 2 ** (-BELIEF_DECAY_INTERVAL / self.adaptive_half_life)
            self.belief_map[self.belief_map > 1e-4] *= decay_factor

    def process_messages(self):
        while self.incoming_messages:
            msg = self.incoming_messages.pop(0)
            self.comm_count += 1
            if msg['type'] == 'map_update':
                self._merge_map_update(msg)

    def step(self, step_num):
        self.step_num = step_num
        self.process_messages()
        self.decay_maps()
        if step_num > 0 and step_num % REASSIGN_INTERVAL == 0:
            self.update_assignments()
