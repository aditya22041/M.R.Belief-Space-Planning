# config.py

# Grid and robot parameters
GRID_SIZE = 100
NUM_ROBOTS = 10
ROBOT_SPEED = 2 # Increased speed to better hunt victims
SENSE_RANGE = 5 # Increased range for better detection
MAX_STEPS = 500

# --- Victim Generation & Motion ---
INITIAL_MOBILE_VICTIMS = 3
INITIAL_DRIFTING_VICTIMS = 3
VICTIM_MOVE_PROBABILITY = 0.3 # Mobile victims have a 30% chance to move each step
VICTIM_DRIFT_VECTOR = (0.1, 0.1)

# --- Dynamic Victim Spawning ---
SPAWN_NEW_VICTIMS = True
VICTIM_SPAWN_INTERVAL = 120
VICTIM_SPAWN_PROBABILITY = 0.5

# --- NEW: Two-Layer Belief System & Communication ---
SCENT_MAP_DECAY = 0.85          # Scent map decays by 15% each step
BELIEF_DECAY_INTERVAL = 10
HIGH_CONFIDENCE_THRESHOLD = 0.80 # Robot's own belief to trigger sending to main belief map
MAP_UPDATE_THRESHOLD = 40       # Increased to reduce communication

# --- "Learning" & Advanced Heuristics ---
ADAPTIVE_DECAY_ENABLED = True
INITIAL_BELIEF_HALF_LIFE = 250
MIN_HALF_LIFE = 150
MAX_HALF_LIFE = 600
ROBOT_HUNT_THRESHOLD = 0.3 # Server's SCENT value to trigger a robot hunt

# --- Region Assignment ---
TILE_SIZE = 15
REASSIGN_INTERVAL = 100

# --- Sensing Noise ---
TRUE_OBSTACLE_RATE = 0.95
FALSE_OBSTACLE_RATE = 0.05
TRUE_POSITIVE_RATE = 0.95 
FALSE_POSITIVE_RATE = 0.05

# --- Environment Generation ---
NUM_RIVERS = 2
RIVER_WIDTH = 2