# belief.py
# -*- coding: utf-8 -*-

import torch
import config
import numpy as np

class BeliefModule:
    def __init__(self, grid_shape):
        # select device
        self.device = torch.device(
            'cuda' if config.USE_GPU and torch.cuda.is_available()
            else 'cpu'
        )
        # tensor of shape (H, W, 3): P(clear), P(obstacle), P(victim)
        self.belief = torch.ones(
            grid_shape[0], grid_shape[1], 3,
            dtype=torch.float32, device=self.device
        ) / 3.0

    def update(self, positions, observations):
        """
        Batched Bayesian update.
        positions: LongTensor[N,2], observations: LongTensor[N] in {0,1,2}
        """
        likelihood = torch.tensor([
            [0.95, 0.05, 0.10],  # obs=clear
            [0.05, 0.95, 0.05],  # obs=obstacle
            [0.05, 0.05, 0.90],  # obs=victim
        ], device=self.device)

        prior = self.belief[positions[:,0], positions[:,1]]       # [N,3]
        like  = likelihood[observations]                          # [N,3]
        unnorm= prior * like                                      # [N,3]
        norm  = unnorm.sum(dim=1, keepdim=True)                   # [N,1]
        post  = torch.where(norm>0, unnorm / norm, prior)         # [N,3]
        # scatter back
        self.belief[positions[:,0], positions[:,1]] = post

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL(p || q) for two probability vectors.
    Adds a tiny epsilon to avoid log(0).
    """
    eps = 1e-9
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))
