# robot.py
import numpy as np
from planner import astar
from config import (SENSE_RANGE, ROBOT_SPEED, TRUE_OBSTACLE_RATE, FALSE_OBSTACLE_RATE,
                    TRUE_POSITIVE_RATE, FALSE_POSITIVE_RATE, MAP_UPDATE_THRESHOLD,
                    HIGH_CONFIDENCE_THRESHOLD, ROBOT_HUNT_THRESHOLD)
import itertools

class Robot:
    def __init__(self, id, pos, grid_shape, server):
        self.id=id; self.pos=np.array(pos,dtype=float); self.grid_shape=grid_shape
        self.server=server; self.speed=ROBOT_SPEED; self.sense_range=SENSE_RANGE
        self.terrain_map=np.full(grid_shape,-1,dtype=np.int8)
        self.explored_map=np.zeros(grid_shape,dtype=bool)
        self.victim_confidence_map=np.zeros(grid_shape,dtype=float)
        self.path=[]; self.target=None; self.newly_sensed_cells=set()
        self.path_cache={}; self.mode="EXPLORE"
        
        # FIX: Re-initialize the frontiers set
        self.frontiers = set()

    def sense(self, true_terrain, true_victims):
        x,y=self.int_pos; H,W=self.grid_shape
        min_r,max_r=max(0,x-self.sense_range),min(H,x+self.sense_range+1)
        min_c,max_c=max(0,y-self.sense_range),min(W,y+self.sense_range+1)
        all_sensed_coords=set((r,c) for r in range(min_r,max_r) for c in range(min_c,max_c))
        if not all_sensed_coords: return
        newly_explored={cell for cell in all_sensed_coords if not self.explored_map[cell]}
        self.newly_sensed_cells.update(newly_explored)
        if newly_explored: self.explored_map[tuple(zip(*newly_explored))]=True
        
        for r,c in newly_explored:
            is_obstacle=true_terrain[r,c]==1
            detected=(is_obstacle and np.random.random()<TRUE_OBSTACLE_RATE) or (not is_obstacle and np.random.random()<FALSE_OBSTACLE_RATE)
            self.terrain_map[r,c]=1 if detected else 0
        
        for r,c in all_sensed_coords:
            if self.terrain_map[r,c]==0:
                is_victim=true_victims[r,c]
                detected=(is_victim and np.random.random()<TRUE_POSITIVE_RATE) or (not is_victim and np.random.random()<FALSE_POSITIVE_RATE)
                if detected:
                    confidence=TRUE_POSITIVE_RATE if is_victim else FALSE_POSITIVE_RATE
                    self.victim_confidence_map[r,c]+= (1-self.victim_confidence_map[r,c])*confidence
        
        # Update frontiers based on newly explored cells
        for r, c in newly_explored:
            self.frontiers.discard((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not self.explored_map[nr, nc]:
                    self.frontiers.add((nr, nc))

    def select_target(self):
        # HUNT mode: Triggered by the server's fast-decaying scent map
        if self.assignment_map is not None:
            # The server object is not available in the robot class.
            # This logic needs to be adapted to use information available to the robot.
            # For now, we'll base the HUNT mode on the robot's own high-confidence beliefs.
            high_belief_indices = np.argwhere(self.victim_confidence_map > ROBOT_HUNT_THRESHOLD)
            my_high_belief_points = [tuple(p) for p in high_belief_indices if self.assignment_map[tuple(p)] == self.id]

            if my_high_belief_points:
                self.mode = "HUNT"
                strongest_scent_pos = max(my_high_belief_points, key=lambda p: self.victim_confidence_map[p])
                self.target = tuple(strongest_scent_pos)
                self.path = []
                return

        self.mode = "EXPLORE"
        # Standard frontier exploration
        if self.assignment_map is None: self.target=self.int_pos; return
        my_frontiers=[f for f in self.frontiers if self.assignment_map[f[0],f[1]]==self.id]
        if not my_frontiers:
            mask=(~self.explored_map)&(self.assignment_map==self.id)
            unexplored_indices=np.argwhere(mask)
            self.target=tuple(unexplored_indices[np.random.randint(len(unexplored_indices))]) if unexplored_indices.size>0 else self.int_pos
            self.path=[]
            return
        self.target=min(my_frontiers,key=lambda f:np.linalg.norm(np.array(f)-self.pos))
        self.path=[]

    def move(self):
        if self.target is None or self.int_pos==self.target: self.path=[]; return
        if not self.path: self.path=astar(self.int_pos,self.target,self.terrain_map,self.path_cache)
        if self.path and self.int_pos==self.path[0]: self.path.pop(0)
        if self.path:
            move_dist=0
            while self.path and move_dist<self.speed:
                next_pos=self.path[0]
                if self.terrain_map[next_pos[0],next_pos[1]]!=1:
                    dist_to_next=np.linalg.norm(np.array(next_pos)-self.pos)
                    if move_dist+dist_to_next<=self.speed:
                        self.pos=np.array(self.path.pop(0),dtype=float); move_dist+=dist_to_next
                    else:
                        direction=(np.array(next_pos)-self.pos)/dist_to_next
                        self.pos+=direction*(self.speed-move_dist); move_dist=self.speed
                else: self.path=[]; self.path_cache.clear(); break

    def communicate(self):
        if len(self.newly_sensed_cells) >= MAP_UPDATE_THRESHOLD:
            positions = list(self.newly_sensed_cells)
            coords=np.array(positions).T
            
            scent_vals = self.victim_confidence_map[tuple(coords)]
            belief_vals = np.where(scent_vals > HIGH_CONFIDENCE_THRESHOLD, scent_vals, 0)

            msg={'type':'map_update','positions':positions,
                 'terrain':self.terrain_map[tuple(coords)].tolist(),
                 'explored':self.explored_map[tuple(coords)].tolist(),
                 'scent':scent_vals, 'belief':belief_vals}
            self.server.incoming_messages.append(msg)
            self.newly_sensed_cells.clear()
            self.victim_confidence_map[tuple(coords)] *= 0.1

    def step(self, true_terrain, true_victims):
        self.sense(true_terrain,true_victims)
        if not self.path: self.select_target()
        self.move()
        self.communicate()

    @property
    def int_pos(self): return tuple(np.round(self.pos).astype(int))