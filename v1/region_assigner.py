# region_assigner.py

import numpy as np
from config import TILE_SIZE

def calculate_tile_uncertainty(tile_confidence, tile_explored):
    """
    Calculate uncertainty for a tile based on victim belief entropy and unexplored cells.
    This function implements the core of the uncertainty metric from the PDF.
    """
    # Entropy for explored cells captures sensor-induced uncertainty.
    p = tile_confidence[tile_explored]
    p = np.clip(p, 1e-9, 1 - 1e-9) # Avoid log(0)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Uncertainty for unexplored cells is considered maximal (1.0 per cell)
    unexplored_count = np.sum(~tile_explored)
    
    total_uncertainty = np.sum(entropy) + unexplored_count
    
    return total_uncertainty / tile_confidence.size if tile_confidence.size > 0 else 0


def assign_square_regions(robots, global_confidence_map, global_explored_map, grid_shape):
    """
    Assigns square regions to robots based on tile uncertainty (entropy/unexplored).
    This implements the greedy, entropy-guided algorithm from the design document.
    """
    H, W = grid_shape
    n_tiles_h = int(np.ceil(H / TILE_SIZE))
    n_tiles_w = int(np.ceil(W / TILE_SIZE))
    
    assignment_map = np.full(grid_shape, -1, dtype=int)
    robot_positions = np.array([r.pos for r in robots])
    
    tiles = []
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            r_start, r_end = i * TILE_SIZE, min((i + 1) * TILE_SIZE, H)
            c_start, c_end = j * TILE_SIZE, min((j + 1) * TILE_SIZE, W)
            
            tile_confidence = global_confidence_map[r_start:r_end, c_start:c_end]
            tile_explored = global_explored_map[r_start:r_end, c_start:c_end]
            
            if not np.all(tile_explored):
                uncertainty = calculate_tile_uncertainty(tile_confidence, tile_explored)
                tiles.append({
                    'coords': (i, j),
                    'uncertainty': uncertainty,
                    'centroid': (r_start + (r_end-r_start)/2, c_start + (c_end-c_start)/2)
                })

    tiles.sort(key=lambda t: t['uncertainty'], reverse=True)
    
    robot_tile_count = {r.id: 0 for r in robots}
    tile_assignments = {} 
    
    for tile in tiles:
        tile_centroid = tile['centroid']
        distances = np.linalg.norm(robot_positions - tile_centroid, axis=1)
        
        nearest_robot_indices = np.argsort(distances)
        best_robot_id = robots[nearest_robot_indices[0]].id
        
        tile_assignments[tile['coords']] = best_robot_id
        robot_tile_count[best_robot_id] += 1
        
    for (i, j), robot_id in tile_assignments.items():
        r_start, r_end = i * TILE_SIZE, min((i + 1) * TILE_SIZE, H)
        c_start, c_end = j * TILE_SIZE, min((j + 1) * TILE_SIZE, W)
        assignment_map[r_start:r_end, c_start:c_end] = robot_id

    unassigned_mask = (assignment_map == -1) & (global_explored_map == False)
    if np.any(unassigned_mask):
        unassigned_indices = np.argwhere(unassigned_mask)
        if unassigned_indices.size > 0:
            diffs = unassigned_indices[None, :, :] - robot_positions[:, None, :]
            sq_dists = (diffs ** 2).sum(axis=2)
            owner_indices = np.array([robots[i].id for i in np.argmin(sq_dists, axis=0)])
            assignment_map[unassigned_mask] = owner_indices

    return assignment_map