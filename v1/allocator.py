# allocator.py

import numpy as np
from config import GRID_SIZE

def compute_assignment(positions, grid_shape=(GRID_SIZE, GRID_SIZE)):
    """
    Partition the grid among robots by assigning each cell to its nearest robot
     Returns a 2D array of shape `grid_shape` where entry [i,j]
    = robot_index that "owns" cell (i,j).
    """
    grid_h, grid_w = grid_shape
    # Build meshgrid of coordinates (H,W,2)
    grid_x, grid_y = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    coords = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)  # (H*W,2)

    # Convert robot positions into array (R,2)
    robot_pos_arr = np.array(positions)  # shape (R,2)

    # Compute squared distances between every robot and every cell:
    #   For each robot r, compute (coords - pos_r)**2 summed over axis 1
    #   Resulting shape = (R, H*W)
    diffs = coords[None, :, :] - robot_pos_arr[:, None, :]  # (R, H*W, 2)
    sq_dists = (diffs ** 2).sum(axis=2)                      # (R, H*W)

    # For each cell (dimension H*W), pick the robot index with min distance
    owner_flat = np.argmin(sq_dists, axis=0)                 # (H*W,)
    assignment_map = owner_flat.reshape(grid_shape)          # (H, W)
    return assignment_map