# communication.py
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree
import torch
import config

class CommunicationManager:
    def __init__(self, robots):
        self.robots = robots
        # initialize snapshots
        for r in robots:
            r.last_shared_belief = r.belief.belief.detach().cpu().numpy().copy()
            r.last_shared_map = r.known_map.copy()
            r.last_shared_victims = len(r.confirmed)

    def step(self, share_factor=None):
        """
        share_factor: δ for partial Bayesian fusion. If None, use config.BELIEF_SHARE_FACTOR.
        """
        δ = share_factor if share_factor is not None else config.BELIEF_SHARE_FACTOR

        # spatial index
        positions = np.array([r.pos for r in self.robots])
        tree = cKDTree(positions)

        for r in self.robots:
            # compute diffs
            belief_curr = r.belief.belief.detach().cpu().numpy()
            belief_diff = np.sum(np.abs(belief_curr - r.last_shared_belief))
            map_diff    = np.count_nonzero(r.known_map != r.last_shared_map)
            victim_diff = len(r.confirmed) - r.last_shared_victims

            if (belief_diff >= config.COMM_BELIEF_THRESH or
                map_diff    >= config.COMM_MAP_THRESH or
                victim_diff >= config.COMM_VICTIM_THRESH):

                neigh_ids = tree.query_ball_point(r.pos, config.COMM_RANGE)
                sent = 0
                for j in neigh_ids:
                    if j == r.id or sent >= config.MAX_COMM_EVENTS:
                        continue
                    tgt = self.robots[j]

                    # partial Bayesian fusion: δ · b1 · b2, then normalize
                    b1 = r.belief.belief
                    b2 = tgt.belief.belief
                    unnorm = δ * b1 * b2
                    denom  = unnorm.sum(dim=2, keepdim=True)
                    merged = torch.where(denom>0, unnorm/denom, b1)

                    # update both agents to the merged belief
                    r.belief.belief = merged.clone()
                    tgt.belief.belief = merged.clone()

                    # merge map & victims as before
                    tgt.known_map = np.maximum(tgt.known_map, r.known_map)
                    tgt.confirmed |= r.confirmed

                    r.comm_count += 1
                    tgt.comm_count += 1
                    sent += 1

                # update snapshots
                r.last_shared_belief = r.belief.belief.detach().cpu().numpy().copy()
                r.last_shared_map    = r.known_map.copy()
                r.last_shared_victims= len(r.confirmed)
