# region_assigner.py

import numpy as np
from config import TILE_SIZE

def calculate_tile_uncertainty(tile_confidence, tile_explored):
    """
    Calculate uncertainty for a tile.
    Higher uncertainty for unexplored cells and for victim probabilities close to 0.5.
    """
    # Entropy for cells with victim confidence
    p = tile_confidence[tile_explored]
    # Avoid log(0) issues
    p = np.clip(p, 1e-9, 1 - 1e-9)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Uncertainty for unexplored cells is considered maximal (1.0)
    unexplored_count = np.sum(~tile_explored)
    
    # Total uncertainty is the sum of entropy and unexplored cell count
    total_uncertainty = np.sum(entropy) + unexplored_count
    
    return total_uncertainty / tile_confidence.size if tile_confidence.size > 0 else 0


def assign_square_regions(robots, global_confidence_map, global_explored_map, grid_shape):
    """
    Assigns square regions to robots based on tile uncertainty (entropy/unexplored).
    Prioritizes high-uncertainty regions and then fills holes.
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
            
            uncertainty = calculate_tile_uncertainty(tile_confidence, tile_explored)
            
            # Only consider tiles that are not fully explored
            if not np.all(tile_explored):
                tiles.append({
                    'coords': (i, j),
                    'uncertainty': uncertainty,
                    'centroid': (r_start + (r_end-r_start)/2, c_start + (c_end-c_start)/2)
                })

    # Sort tiles by uncertainty, descending
    tiles.sort(key=lambda t: t['uncertainty'], reverse=True)
    
    # Greedily assign high-uncertainty tiles to the nearest robot
    tile_assignments = {} # (i,j) -> robot_id
    for tile in tiles:
        tile_centroid = tile['centroid']
        # Find distance from tile centroid to each robot
        distances = np.linalg.norm(robot_positions - tile_centroid, axis=1)
        nearest_robot_id = np.argmin(distances)
        tile_assignments[tile['coords']] = nearest_robot_id
        
    # Create the assignment map from tile assignments
    for (i, j), robot_id in tile_assignments.items():
        r_start, r_end = i * TILE_SIZE, (i + 1) * TILE_SIZE
        c_start, c_end = j * TILE_SIZE, (j + 1) * TILE_SIZE
        assignment_map[r_start:r_end, c_start:c_end] = robot_id

    # --- Hole Filling Pass ---
    # Identify small unassigned regions and assign them to the majority neighbor
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            if tile_assignments.get((i, j)) is None:
                neighbor_owners = []
                # Check 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0: continue
                        owner = tile_assignments.get((i + di, j + dj))
                        if owner is not None:
                            neighbor_owners.append(owner)
                
                if neighbor_owners:
                    # Assign to the most common neighboring robot
                    majority_owner = max(set(neighbor_owners), key=neighbor_owners.count)
                    r_start, r_end = i * TILE_SIZE, (i + 1) * TILE_SIZE
                    c_start, c_end = j * TILE_SIZE, (j + 1) * TILE_SIZE
                    assignment_map[r_start:r_end, c_start:c_end] = majority_owner

    return assignment_map