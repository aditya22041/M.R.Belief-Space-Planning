import numpy as np

def generate_terrain(grid_shape):
    """Generate a grid with obstacles (20% density)."""
    grid = np.zeros(grid_shape, dtype=np.int8)
    grid[np.random.rand(*grid_shape) < 0.2] = 1  # 1 = obstacle
    return grid

def generate_victims(grid_shape, terrain):
    """Place victims in 5% of free cells."""
    victims = np.zeros(grid_shape, dtype=bool)
    free_cells = np.where(terrain == 0)
    num_victims = int(0.05 * len(free_cells[0]))
    indices = np.random.choice(len(free_cells[0]), num_victims, replace=False)
    victims[free_cells[0][indices], free_cells[1][indices]] = True
    return victims