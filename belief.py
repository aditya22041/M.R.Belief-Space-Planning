# belief.py

import numpy as np
from config import TRUE_POSITIVE_RATE, FALSE_POSITIVE_RATE, FALSE_OBSTACLE_RATE

class BeliefModule:
    """
    Maintains, for each cell, a probability vector [P(free), P(obstacle), P(victim)].
    Updates are done with Bayes’ rule in NumPy.
    """

    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.belief = np.ones((grid_shape[0], grid_shape[1], 3), dtype=np.float64) / 3.0
        self.uncertain_mask = np.zeros(grid_shape, dtype=bool)
        self.update_uncertain_mask()

    def update_uncertain_mask(self):
        """
        A cell is ‘uncertain’ if 0.1 < P(victim) < 0.8.
        """
        p_v = self.belief[:, :, 2]
        self.uncertain_mask = (p_v > 0.1) & (p_v < 0.8)

    def bayesian_update(self, positions, observations):
        """
        Given a batch of sensed (i,j) positions and their observation codes:
          - 0 = “free, no victim detected”
          - 1 = “obstacle detected”
          - 2 = “victim detected”
        We do a vectorized Bayes update for those cells.
        """
        if len(positions) == 0:
            return

        pos_arr = np.array(positions, dtype=int)   # shape (N,2)
        obs_arr = np.array(observations, dtype=int)  # shape (N,)

        # 1) obs == 1 ⇒ definitely obstacle ⇒ belief = [0,1,0]
        mask_obst = (obs_arr == 1)
        if np.any(mask_obst):
            idxs = pos_arr[mask_obst]
            self.belief[idxs[:,0], idxs[:,1], :] = 0.0
            self.belief[idxs[:,0], idxs[:,1], 1] = 1.0

        # 2) obs == 0 ⇒ “free/no‐victim”
        mask_free = (obs_arr == 0)
        if np.any(mask_free):
            idxs = pos_arr[mask_free]
            prior = self.belief[idxs[:,0], idxs[:,1], :]    # shape (n,3)

            # Likelihoods:
            #   P(obs=0 | free)     = 1 - FALSE_OBSTACLE_RATE
            #   P(obs=0 | obstacle) = FALSE_OBSTACLE_RATE
            #   P(obs=0 | victim)   = FALSE_POSITIVE_RATE
            like = np.array([
                1.0 - FALSE_OBSTACLE_RATE,
                FALSE_OBSTACLE_RATE,
                FALSE_POSITIVE_RATE
            ]).reshape((1,3))  # shape (1,3)

            posterior = prior * like                             # shape (n,3)
            norms = posterior.sum(axis=1, keepdims=True).clip(min=1e-6)
            new_p = posterior / norms                            # shape (n,3)
            self.belief[idxs[:,0], idxs[:,1], :] = new_p

        # 3) obs == 2 ⇒ “victim detected”
        mask_vic = (obs_arr == 2)
        if np.any(mask_vic):
            idxs = pos_arr[mask_vic]
            prior = self.belief[idxs[:,0], idxs[:,1], :]        # shape (n,3)

            # Likelihoods:
            #   P(obs=2 | victim)   = TRUE_POSITIVE_RATE
            #   P(obs=2 | free)     = 1 - TRUE_POSITIVE_RATE
            #   P(obs=2 | obstacle) = 0
            like = np.array([
                1.0 - TRUE_POSITIVE_RATE,
                0.0,
                TRUE_POSITIVE_RATE
            ]).reshape((1,3))

            posterior = prior * like                             # shape (n,3)
            norms = posterior.sum(axis=1, keepdims=True).clip(min=1e-6)
            new_p = posterior / norms                            # shape (n,3)
            self.belief[idxs[:,0], idxs[:,1], :] = new_p

        # Update uncertain mask after each update
        self.update_uncertain_mask()

    def merge_external_belief(self, positions, external_probs):
        """
        When a robot receives another’s belief for positions:
        We simply do (prior + external) / sum to re‐normalize (equal‐weight fusion).
        """
        if len(positions) == 0:
            return

        pos_arr = np.array(positions, dtype=int)                   # (N,2)
        ext_arr = np.array(external_probs, dtype=float)            # (N,3)
        prior = self.belief[pos_arr[:,0], pos_arr[:,1], :]         # (N,3)

        combined = prior + ext_arr                                  # (N,3)
        norms = combined.sum(axis=1, keepdims=True).clip(min=1e-6)  # (N,1)
        fused = combined / norms                                    # (N,3)
        self.belief[pos_arr[:,0], pos_arr[:,1], :] = fused

        self.update_uncertain_mask()
