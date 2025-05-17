# multi_robot_bsp/allocator.py
"""
Task allocation and frontier partitioning strategies:
  - CVT (Centroidal Voronoi Tessellation via Lloyd's algorithm)
  - Voronoi assignment based on robot positions
  - Auction-based greedy assignment

Previous modules:
  - config.py
  - terrain.py
  - belief.py
  - communication.py
  - planner.py
"""
import numpy as np
import random
import config

# --- CVT Partitioning ---
def cvt_partition(frontiers, num_robots, max_iter=20):
    """
    Partition frontier points into num_robots clusters via Lloyd's algorithm.
    Returns centers (array num_robots x 2) and clusters dict.
    """
    pts = np.array(frontiers)
    n_pts = len(pts)
    if n_pts == 0:
        return np.zeros((num_robots, 2)), {i: [] for i in range(num_robots)}
    # initialize centers randomly
    idx = np.random.choice(n_pts, num_robots, replace=n_pts < num_robots)
    centers = pts[idx].astype(float)
    for _ in range(max_iter):
        # assign
        dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(num_robots):
            members = pts[labels == i]
            if len(members) > 0:
                new_centers[i] = members.mean(axis=0)
            else:
                # if no members, reinitialize center
                new_centers[i] = pts[random.randrange(n_pts)]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    clusters = {i: [tuple(pt) for pt in pts[labels == i]] for i in range(num_robots)}
    return centers, clusters

# --- Voronoi Partitioning ---
def voronoi_partition(frontiers, robot_positions):
    """
    Assign each frontier to the nearest robot via Euclidean distance.
    Returns robot_positions (clusters keyed by index).
    """
    pts = np.array(frontiers)
    robots = np.array(robot_positions)
    if len(pts) == 0:
        return {i: [] for i in range(len(robot_positions))}
    # compute distance matrix
    dists = np.linalg.norm(pts[:, None, :] - robots[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    clusters = {i: [] for i in range(len(robot_positions))}
    for pt, lbl in zip(pts, labels):
        clusters[int(lbl)].append(tuple(pt))
    return clusters

# --- Auction-based Partitioning ---
def auction_partition(frontiers, robot_positions):
    """
    Greedy auction: robots sequentially claim nearest frontier.
    Ensures balanced assignment.
    """
    remaining = set(frontiers)
    num_r = len(robot_positions)
    clusters = {i: [] for i in range(num_r)}
    # round-robin auction
    idx = 0
    while remaining:
        robot = robot_positions[idx % num_r]
        # bid = nearest frontier
        best = min(remaining, key=lambda p: abs(p[0]-robot[0]) + abs(p[1]-robot[1]))
        clusters[idx % num_r].append(best)
        remaining.remove(best)
        idx += 1
    return clusters

# --- Unit Tests ---
if __name__ == "__main__":
    # generate sample frontiers
    fr = [(random.randint(0,50), random.randint(0,50)) for _ in range(100)]
    robots = [(0,0), (50,50), (0,50), (50,0)]

    # CVT
    centers, clusters_cvt = cvt_partition(fr, len(robots), max_iter=10)
    total = sum(len(v) for v in clusters_cvt.values())
    assert total == len(fr), f"CVT assigned {total}/{len(fr)}"

    # Voronoi
    clusters_vor = voronoi_partition(fr, robots)
    total_v = sum(len(v) for v in clusters_vor.values())
    assert total_v == len(fr), f"Voronoi assigned {total_v}/{len(fr)}"

    # Auction
    clusters_auc = auction_partition(fr, robots)
    total_a = sum(len(v) for v in clusters_auc.values())
    assert total_a == len(fr), f"Auction assigned {total_a}/{len(fr)}"

    print("allocator.py tests passed.")
