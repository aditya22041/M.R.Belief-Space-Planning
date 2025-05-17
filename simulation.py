# simulation.py
import time
import argparse
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import config
from terrain import TerrainGenerator
from belief import BeliefModule, kl_divergence
from communication import CommunicationManager
from planner import DStarLitePlanner, RRTStarPlanner
from allocator import cvt_partition, voronoi_partition, auction_partition

ALLOC_INTERVAL = getattr(config, 'ALLOC_INTERVAL', 5)
COMM_INTERVAL  = config.COMM_INTERVAL

class RobotAgent:
    def __init__(self, robot_id, start_pos, grid_shape):
        self.id             = robot_id
        self.pos            = tuple(start_pos)
        self.belief         = BeliefModule(grid_shape)
        self.known_map      = np.full(grid_shape, -1, dtype=np.int8)
        self.confirmed      = set()
        self.new_observed   = set()
        self.new_confirmed  = set()
        self.comm_count     = 0
        self.path           = []
        self.target         = self.pos

    def sense(self, terrain, victims):
        r = config.SENSOR_RADIUS
        h, w = terrain.shape
        obs_pos, obs_code = [], []
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                i, j = self.pos[0]+dx, self.pos[1]+dy
                if not (0<=i<h and 0<=j<w): continue
                truth = 1 if terrain[i,j]==1 else (2 if (i,j) in victims else 0)
                if truth==2:
                    obs = 2 if random.random()<config.DETECTION_PROB else 0
                elif truth==0:
                    obs = 2 if random.random()<config.FALSE_POS_PROB else 0
                else:
                    obs = 1
                prev = self.known_map[i,j]
                val  = 1 if obs==1 else 0
                self.known_map[i,j] = val
                if prev<0:
                    self.new_observed.add((i,j))
                obs_pos.append((i,j)); obs_code.append(obs)

        import torch
        coords = np.array(obs_pos, dtype=int)
        pos_t  = torch.from_numpy(coords).long().to(self.belief.device)
        obs_t  = torch.tensor(obs_code, dtype=torch.long, device=self.belief.device)
        self.belief.update(pos_t, obs_t)

    def confirm_by_posterior(self, thresh, true_victims):
        post = self.belief.belief.detach().cpu().numpy()[...,2]
        for cell in zip(*np.where(post >= thresh)):
            if cell in true_victims and cell not in self.confirmed:
                self.confirmed.add(cell)
                self.new_confirmed.add(cell)

    def plan(self, terrain):
        if self.pos == self.target or not self.path:
            if config.PLANNER_TYPE == 'DSTAR_LITE':
                planner = DStarLitePlanner(terrain, self.pos, self.target)
                self.path = planner.plan()
            else:
                def free_chk(pt):
                    x, y = map(int, pt)
                    return (0<=x<terrain.shape[0] and 0<=y<terrain.shape[1]
                            and terrain[x,y]==0)
                bounds = ((0,0),(terrain.shape[0],terrain.shape[1]))
                self.path = RRTStarPlanner(self.pos, self.target,
                                           free_chk, bounds).plan()

    def move(self):
        if len(self.path) > 1:
            self.pos  = tuple(self.path[1])
            self.path = self.path[1:]
        else:
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = self.pos[0]+dx, self.pos[1]+dy
                if (0<=ni<self.known_map.shape[0]
                        and 0<=nj<self.known_map.shape[1]
                        and self.known_map[ni,nj]==0):
                    self.pos = (ni,nj)
                    break


def run_simulation(args):
    # Generate map and victims
    tg      = TerrainGenerator(seed=args.seed)
    terrain, victims = tg.generate()
    shape   = terrain.shape

    # Plot initial map
    plt.figure(figsize=(6,6))
    plt.imshow(terrain, cmap='gray_r', origin='lower')
    vx, vy = zip(*victims)
    plt.scatter(vy, vx, facecolors='none', edgecolors='red', s=50, label='True victims')
    plt.title('Initial Terrain (white=free, black=obstacle) & True Victims')
    plt.legend(); plt.show()

    # Initialize robots
    free    = list(zip(*np.where(terrain==0)))
    random.shuffle(free)
    robots  = [RobotAgent(i, free[i%len(free)], shape)
               for i in range(args.num_robots)]
    comm    = CommunicationManager(robots)

    total_cells   = shape[0]*shape[1]
    coverage_ts   = []
    comm_ts       = []
    confirmed_ts  = []
    comm_cum_ts   = []

    t0 = time.time()
    global_map = np.full(shape, -1, dtype=np.int8)

    # plateau tracking
    plateau_count = 0
    last_alloc_cov = -1

    for step in range(args.max_steps):
        # 1) sense & belief update
        for r in robots: r.sense(terrain, victims)

        # 2) confirm victims
        for r in robots:
            r.confirm_by_posterior(args.confirm_belief_thresh, set(victims))

        # 3) update shared map & coverage
        for r in robots:
            for cell in r.new_observed:
                global_map[cell] = r.known_map[cell]
            r.new_observed.clear()
        seen     = np.count_nonzero(global_map >= 0)
        coverage = 100 * seen / total_cells
        coverage_ts.append(coverage)

        # 4) allocation & planning
        if step % ALLOC_INTERVAL == 0:
            # detect plateau
            if coverage <= last_alloc_cov:
                plateau_count += 1
            else:
                plateau_count = 0
            last_alloc_cov = coverage

            unk   = (global_map == -1).astype(int)
            kern  = np.array([[0,1,0],[1,0,1],[0,1,0]])
            neigh = convolve2d(unk, kern, mode='same', boundary='fill', fillvalue=0)
            frontiers     = list(zip(*np.where((global_map==0)&(neigh>0))))
            unknown_cells = list(zip(*np.where(global_map==-1)))

            # always explorers: half the robots
            num_explorers = len(robots)//2
            explorers     = set(range(num_explorers))

            # if plateau or no frontiers, random targets for all
            if plateau_count >= 2 or not frontiers:
                for r in robots:
                    if unknown_cells:
                        r.target = random.choice(unknown_cells)
                    r.plan(terrain)
                plateau_count = 0
            else:
                # partition actual frontiers
                if config.COVERAGE_STRATEGY=='CVT':
                    _, clusters = cvt_partition(frontiers, len(robots))
                elif config.COVERAGE_STRATEGY=='Voronoi':
                    clusters = voronoi_partition(frontiers, [r.pos for r in robots])
                else:
                    clusters = auction_partition(frontiers, [r.pos for r in robots])

                for r in robots:
                    # explorers always random
                    if r.id in explorers and unknown_cells:
                        r.target = random.choice(unknown_cells)
                    else:
                        targets = clusters.get(r.id, []) or unknown_cells
                        r.target = min(targets, key=lambda p: abs(p[0]-r.pos[0]) + abs(p[1]-r.pos[1]))
                    r.plan(terrain)

        # 5) move
        for r in robots: r.move()

        # 6) communication
        prev_comm = sum(r.comm_count for r in robots)
        if step % COMM_INTERVAL == 0 and any(r.new_confirmed for r in robots):
            comm.step()
        comm_delta = sum(r.comm_count for r in robots) - prev_comm
        comm_ts.append(comm_delta)

        # 7) stats
        total_conf = len(set().union(*(r.confirmed for r in robots)))
        confirmed_ts.append(total_conf)
        comm_cum_ts.append(sum(r.comm_count for r in robots))

        # 8) KL convergence
        maps = [r.belief.belief.detach().cpu().numpy()[...,2] for r in robots]
        max_kl = 0
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                max_kl = max(max_kl, kl_divergence(maps[i].ravel(), maps[j].ravel()))
        if max_kl < config.KL_CONV_THRESH:
            print(f"Beliefs converged at step {step+1} (max KL={max_kl:.2e})")
            break

    # final report
    elapsed   = time.time() - t0
    confirmed = set().union(*(r.confirmed for r in robots))
    comms     = [r.comm_count for r in robots]
    positions = [r.pos for r in robots]

    print("\n===== Simulation Report =====")
    print(f"Elapsed     : {elapsed:.2f}s   Steps: {step+1}")
    print(f"Coverage    : {coverage_ts[-1]:.2f}%  ({seen}/{total_cells})")
    print(f"Victims hit : {len(confirmed)}/{len(victims)}")
    print(f"Confirmed   : {sorted((int(x),int(y)) for x,y in confirmed)}")
    print(f"Comms       : min={min(comms)}, max={max(comms)}, avg={np.mean(comms):.2f}")
    print(f"Positions   : {[(int(x),int(y)) for x,y in positions]}")
    print("============================\n")

    # plots
    plt.figure(figsize=(6,3)); plt.plot(coverage_ts, label='Coverage %')
    plt.xlabel('Step'); plt.ylabel('Coverage %'); plt.title('Coverage over Time'); plt.grid(True); plt.legend(); plt.show()

    plt.figure(figsize=(6,3)); plt.plot(comm_ts, label='Comm Δ')
    plt.xlabel('Step'); plt.ylabel('# Comm events'); plt.title('Communication Events per Interval'); plt.grid(True); plt.legend(); plt.show()

    new_per_step = np.diff([0] + confirmed_ts)
    plt.figure(figsize=(6,3)); plt.bar(range(len(new_per_step)), new_per_step, color='tab:orange', label='New victims')
    plt.xlabel('Step'); plt.ylabel('# New confirmed'); plt.title('Victims Discovered per Step'); plt.grid(True); plt.legend(); plt.show()

    plt.figure(figsize=(6,3)); plt.plot(comm_cum_ts, label='Total Comm so far')
    plt.xlabel('Step'); plt.ylabel('Total comm events'); plt.title('Cumulative Communication Load'); plt.grid(True); plt.legend(); plt.show()

    # final heatmap overlay
    avg_map = np.mean(maps, axis=0)
    plt.figure(figsize=(6,6)); plt.imshow(avg_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Avg P(victim)')
    plt.scatter([c[1] for c in confirmed], [c[0] for c in confirmed], c='red', marker='x', label='Confirmed')
    plt.scatter([v[1] for v in victims], [v[0] for v in victims], facecolors='none', edgecolors='white', label='True victims')
    plt.title('Fleet-wide Average Victim Probability\n(white circles=true, red X=confirmed)')
    plt.legend(loc='upper right'); plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-robots', type=int, default=config.NUM_ROBOTS)
    parser.add_argument('--max-steps',  type=int, default=200)
    parser.add_argument('--confirm-belief-thresh', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    run_simulation(parser.parse_args())
