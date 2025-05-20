# simulation.py

import numpy as np
import matplotlib.pyplot as plt
import time
from environment import Environment
from robot import Robot
from communication import CommunicationManager
from config import GRID_SIZE, NUM_ROBOTS, MAX_STEPS

def run_simulation():
    start_time = time.time()

    # 1) Initialize environment & robots
    env = Environment(GRID_SIZE)
    free_cells = np.argwhere(env.reachable & (env.terrain_map == 0))
    num_free = len(free_cells)
    actual_robots = min(NUM_ROBOTS, num_free)
    if actual_robots < NUM_ROBOTS:
        print(f"Warning: Only {actual_robots} robots placed (free cells < requested).")
    if num_free == 0:
        print("Error: No free cells to place robots. Exiting.")
        return

    rng = np.random.default_rng()
    chosen_idx = rng.choice(num_free, size=actual_robots, replace=False)
    robot_positions = free_cells[chosen_idx]
    robots = [
        Robot(i, tuple(robot_positions[i]), (GRID_SIZE, GRID_SIZE))
        for i in range(actual_robots)
    ]

    comm = CommunicationManager(robots)

    # Initial plot: obstacles=black, free=white, plus true victims
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(1 - env.terrain_map, cmap='gray', origin='lower')
    tv = np.argwhere(env.victim_map)
    if len(tv) > 0:
        ax.scatter(tv[:,1], tv[:,0], c='green', marker='x', label='True Victims')
    ax.legend()
    ax.set_title("Initial Terrain (white=free, black=obstacle) & True Victims (green x)")
    plt.savefig('initial_map.png')
    plt.show()

    select_t = sense_t = move_t = comm_t = 0.0

    # Main simulation loop
    for step in range(MAX_STEPS):
        for r in robots:
            # Target selection
            t0 = time.time()
            r.select_target(step, robots)
            select_t += time.time() - t0

            # Sensing
            t1 = time.time()
            r.sense(env.terrain_map, env.victim_map)
            sense_t += time.time() - t1

            # Movement & broadcast
            t2 = time.time()
            r.move(comm)
            move_t += time.time() - t2

        # Communication step
        t3 = time.time()
        comm.step()
        comm_t += time.time() - t3

    # Build global maps
    global_explored = np.any([r.explored_map for r in robots], axis=0)
    global_detected = np.any([
        (r.confidence_map.belief[:,:,2] > 0.8) & r.victim_map
        for r in robots
    ], axis=0)

    print("\n=== Simulation Summary ===")
    print("\nOriginal Terrain Map (top-left 10×10):")
    print(env.terrain_map[:10, :10])

    for r in robots:
        explored_area = np.sum(r.explored_map)
        dets = np.argwhere((r.confidence_map.belief[:,:,2] > 0.8) & r.victim_map)
        locs = [(int(x), int(y)) for x,y in dets]
        print(f"\nRobot {r.id}:")
        print(f"  Final Position: ({r.pos[0]}, {r.pos[1]})")
        print(f"  Area Explored: {explored_area} cells")
        print(f"  Explored Map (top-left 10×10):")
        print(r.explored_map[:10, :10].astype(int))
        print(f"  Detected Victims: {locs if locs else 'None'}")
        print(f"  Communication Count: {r.comm_count}")

    # Victim detection metrics
    true_set = set(map(tuple, np.argwhere(env.victim_map).tolist()))
    det_set  = set(map(tuple, np.argwhere(global_detected).tolist()))
    correct    = len(true_set & det_set)
    false_pos  = len(det_set - true_set)
    missed     = len(true_set - det_set)
    prec = correct / (correct + false_pos) if (correct + false_pos) > 0 else 0.0
    rec  = correct / len(true_set) if len(true_set) > 0 else 0.0

    print("\nVictim Detection Metrics:")
    print(f"  True Victims: {len(true_set)}")
    print(f"  Correct Detections: {correct}")
    print(f"  False Positives: {false_pos}")
    print(f"  Missed Victims: {missed}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall: {rec:.2f}")

    # Final plots
    fig, axes = plt.subplots(1, 3, figsize=(24,8))

    # 1) Terrain + Explored + Robots
    axes[0].imshow(1 - env.terrain_map, cmap='gray', origin='lower')
    axes[0].imshow(global_explored, cmap='Blues', alpha=0.5, origin='lower')
    for i, r in enumerate(robots):
        axes[0].plot(r.pos[1], r.pos[0], 'bo', label='Robot' if i == 0 else None)
    axes[0].legend()
    axes[0].set_title("Terrain + Explored + Final Robot Positions")

    # 2) True vs Detected Victims
    axes[1].imshow(1 - env.terrain_map, cmap='gray', origin='lower')
    if len(tv) > 0:
        axes[1].scatter(tv[:,1], tv[:,0], c='green', marker='x', label='True Victims')
    dv = np.argwhere(global_detected)
    if dv.size > 0:
        axes[1].scatter(dv[:,1], dv[:,0], c='red', marker='o', label='Detected')
    axes[1].legend()
    axes[1].set_title("True (green x) vs Detected (red circle) Victims")

    # 3) Robot Positions & “Long” Communication Counts
    axes[2].set_xlim(-1, GRID_SIZE)
    axes[2].set_ylim(-1, GRID_SIZE)
    for i, r in enumerate(robots):
        axes[2].plot(r.pos[1], r.pos[0], 'bo', label='Robot' if i == 0 else None)
        if r.comm_count > 0:
            axes[2].text(r.pos[1], r.pos[0], f"{r.comm_count}", color='purple')
    axes[2].legend()
    axes[2].set_title("Robot Positions & Communication Counts")

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()

    end_time = time.time()
    print("\nPerformance Breakdown:")
    print(f"  Target Selection Time: {select_t:.2f}s")
    print(f"  Sensing Time: {sense_t:.2f}s")
    print(f"  Movement Time: {move_t:.2f}s")
    print(f"  Communication Time: {comm_t:.2f}s")
    print(f"Total Simulation Time: {end_time - start_time:.2f}s")
    print(f"Total Free Cells: {num_free}")
    print(f"Robots Placed: {actual_robots}")

if __name__ == "__main__":
    run_simulation()
