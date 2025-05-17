# terrain.py
"""
Map generation utilities: scalable obstacle & victim placement.
"""
import numpy as np
import random
import config

class TerrainGenerator:
    def __init__(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        self.size = config.MAP_SIZE
        self.obstacle_density = config.OBSTACLE_DENSITY

    def generate(self):
        # 0 = free, 1 = obstacle
        grid = (np.random.rand(self.size, self.size) < self.obstacle_density).astype(np.int8)

        # place victims on random free cells
        free_cells = np.column_stack(np.where(grid == 0))
        free_list = free_cells.tolist()
        random.shuffle(free_list)
        num = min(len(free_list), config.NUM_VICTIMS)
        victims = [tuple(pt) for pt in free_list[:num]]

        return grid, victims

if __name__ == "__main__":
    tg = TerrainGenerator(seed=42)
    grid, victims = tg.generate()
    assert grid.shape == (config.MAP_SIZE, config.MAP_SIZE)
    assert set(np.unique(grid)).issubset({0, 1})
    assert len(victims) == min(config.NUM_VICTIMS, (grid == 0).sum())
    print("terrain.py tests passed.")
