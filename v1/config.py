# config.py

# Grid and robot parameters
GRID_SIZE = 100               # Side length of the square grid
NUM_ROBOTS = 10               # Number of robots
ROBOT_SPEED = 2               # Cells moved per step (increase to move faster)
SENSE_RANGE = 2               # Manhattan‐radius of sensing (2 → 5×5 window). Adjustable.
ALLOC_INTERVAL = 40           # Steps between re‐computing assignment partition
MAX_STEPS = 300               # Total simulation steps (increased for more thorough exploration)

# New Tiling strategy parameter
TILE_SIZE = 10                # The side length of square regions for assignment

# Sensing noise (terrain/victim)
TRUE_OBSTACLE_RATE = 0.95     # P(detect obstacle | obstacle)
FALSE_OBSTACLE_RATE = 0.05    # P(detect obstacle | free)
TRUE_POSITIVE_RATE = 0.90     # P(detect victim | victim present)
FALSE_POSITIVE_RATE = 0.10    # P(detect victim | no victim)

# Victim Detection
VICTIM_CONFIDENCE_THRESHOLD = 0.5 # Confidence threshold for a robot to log a victim

# Communication thresholds
MAP_UPDATE_THRESHOLD = 50     # Cells of newly explored data before sending a map update to the server

OBSTACLE_THRESHOLD= 10
VICTIM_DENSITY= 0.005 