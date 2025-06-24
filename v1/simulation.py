# simulation.py

import numpy as np
import matplotlib.pyplot as plt
import time
from environment import Environment
from robot import Robot
from central_server import CentralServer
from config import GRID_SIZE, NUM_ROBOTS, MAX_STEPS

def run_simulation():
    start_time = time.time()

    # 1) Initialize environment, central server, and robots
    env = Environment(GRID_SIZE)
    free_cells = np.argwhere(env.reachable)
    
    if len(free_cells) < NUM_ROBOTS:
        print(f"Warning: Not enough free cells ({len(free_cells)}) for {NUM_ROBOTS} robots.")
        return

    rng = np.random.default_rng()
    start_indices = rng.choice(len(free_cells), size=NUM_ROBOTS, replace=False)
    robot_positions = free_cells[start_indices]

    server = CentralServer((GRID_SIZE, GRID_SIZE), [])
    robots = [
        Robot(i, tuple(pos), (GRID_SIZE, GRID_SIZE), server)
        for i, pos in enumerate(robot_positions)
    ]
    server.robots = {r.id: r for r in robots}
    
    # Initial assignment
    server.update_assignments()

    print(f"Starting simulation with {NUM_ROBOTS} robots for {MAX_STEPS} steps.")
    
    # Main simulation loop
    for step in range(MAX_STEPS):
        if step > 0 and step % 50 == 0:
            print(f"--- Step {step}/{MAX_STEPS} ---")

        for r in robots:
            r.step(env.terrain_map, env.victim_map)

        server.step(step)
    
    print("\n=== Simulation Summary ===")
    
    # Final metrics
    true_victims_set = set(map(tuple, np.argwhere(env.victim_map)))
    avg_confidence_at_victim_locs = np.mean(server.global_confidence_map[env.victim_map])
    avg_confidence_at_free_locs = np.mean(server.global_confidence_map[~env.victim_map & env.reachable])
    
    print("\nProbabilistic Mapping Metrics:")
    print(f"  Avg. Final Confidence at True Victim Locations: {avg_confidence_at_victim_locs:.3f}")
    print(f"  Avg. Final Confidence at Non-Victim Locations: {avg_confidence_at_free_locs:.3f}")
    
    explored_area_percent = np.sum(server.global_explored_map) / np.sum(env.reachable) * 100
    print(f"\nTotal Explored Area: {explored_area_percent:.2f}% of reachable cells.")
    print(f"Total Communications: {server.comm_count} messages.")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds.")

    # Final plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Simulation Results (Entropy-Based Square Region Exploration)")

    # 1) Region Assignments
    axes[0].imshow(server.global_terrain_map, cmap='gray_r', origin='lower', interpolation='none')
    axes[0].imshow(server.robots[0].assignment_map, cmap='tab20', alpha=0.5, origin='lower', interpolation='none')
    axes[0].set_title("Final Region Assignments")

    # 2) Victim Confidence Heatmap
    im = axes[1].imshow(server.global_confidence_map, cmap='hot', origin='lower', interpolation='none')
    axes[1].set_title("Final Victim Confidence Heatmap")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Ground Truth
    axes[2].imshow(1 - env.terrain_map, cmap='gray', origin='lower', interpolation='none')
    true_victim_pos = np.argwhere(env.victim_map)
    axes[2].scatter(true_victim_pos[:, 1], true_victim_pos[:, 0], c='lime', marker='x', s=100, label='True Victims')
    axes[2].set_title("Ground Truth Victims")
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('simulation_results_v_entropy.png')
    plt.show()

if __name__ == "__main__":
    run_simulation()