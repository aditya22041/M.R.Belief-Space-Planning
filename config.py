# config.py
# -*- coding: utf-8 -*-

import os
import logging

# Logging
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "mr_bsp.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MRBSP")

# Map dimensions & density
MAP_SIZE = 100                # 1000x1000 grid
OBSTACLE_DENSITY = 0.2         # fraction of obstacles
NUM_VICTIMS = 50

# Robot fleet
NUM_ROBOTS = 10
SENSOR_RADIUS = 5              # Manhattan radius sensing
COMM_RANGE = 100               # communication radius
MAX_COMM_EVENTS = 5            # max messages per step per leader

# Planner & Allocation strategies
PLANNER_TYPE = "DSTAR_LITE"    # Options: DSTAR_LITE, RRT_STAR
COVERAGE_STRATEGY = "CVT"      # Options: CVT, Voronoi, Auction

# Sensor model
DETECTION_PROB = 0.9           # P(detect | victim present)
FALSE_POS_PROB = 0.05          # P(false detect | no victim)

# Performance
USE_GPU = True
BATCH_SIZE = 512               # for batched belief updates

# Communication thresholds
COMM_BELIEF_THRESH = 10000.0     # L1 belief-change threshold to trigger comm
COMM_MAP_THRESH = 50           # number of newly-explored cells to trigger comm
COMM_VICTIM_THRESH = 1         # number of new confirmed victims to trigger comm
COMM_INTERVAL = 10             # steps between communication gating

# Belief‐merge & convergence
BELIEF_SHARE_FACTOR = 0.5      # δ in [0,1] for partial Bayesian fusion
KL_CONV_THRESH      = 1e-3     # stop when max pairwise KL < this

if __name__ == "__main__":
    # Basic sanity tests
    assert isinstance(MAP_SIZE, int) and MAP_SIZE > 0
    assert 0 < OBSTACLE_DENSITY < 1
    assert NUM_ROBOTS > 0
    assert 0 <= DETECTION_PROB <= 1
    assert 0 <= FALSE_POS_PROB <= 1
    print("config.py tests passed.")
