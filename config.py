# config.py

# Grid and robot parameters
GRID_SIZE = 100               # Side length of the square grid
NUM_ROBOTS = 10               # Number of robots
ROBOT_SPEED = 2               # Cells moved per step (increase to move faster)
SENSE_RANGE = 2               # Manhattan‐radius of sensing (2 → 5×5 window). Adjustable.
ALLOC_INTERVAL = 40           # Steps between re‐computing assignment partition
MAX_STEPS = 300               # Total simulation steps

# Sensing noise (terrain/victim)
TRUE_OBSTACLE_RATE = 0.95     # P(detect obstacle | obstacle)
FALSE_OBSTACLE_RATE = 0.05    # P(detect obstacle | free)
TRUE_POSITIVE_RATE = 0.90     # P(detect victim | victim present)
FALSE_POSITIVE_RATE = 0.10    # P(detect victim | no victim)

# Communication thresholds
SHARE_THRESHOLD = 100          # Cells of newly explored data before broadcast share
COMM_RANGE = 20               # Euclidean communication radius (in cells)
REQUEST_PROB = 0.10           # P(each step to broadcast a confirmation request)
MAX_SHARE = 20                # Max cells to share per large communication

# Environment parameters
OBSTACLE_THRESHOLD = 0.65     # Perlin‐noise threshold to turn a cell into obstacle
VICTIM_DENSITY = 0.005         # Fraction of reachable cells containing a victim
