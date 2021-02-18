import random
import time
from typing import Tuple

import numpy as np
import pyglet
from numpy import ndarray
from pyglet import shapes
from pyglet.graphics import Batch
from pyglet.gl import *


def main():
    window = SimWindow()
    window.sim_start()


class SimWindow(pyglet.window.Window):
    def __init__(self):
        super(SimWindow, self).__init__(600, 600)
        self.sim_dt = 1.0 / 60.0

        state_array = PLL_State.solved_PLL_state()
        n = 25
        P2_10 = create_permutation_matrix(2, 10, n)
        P10_14 = create_permutation_matrix(10, 14, n)
        P10_22 = create_permutation_matrix(10, 22, n)
        P = P10_14 @ P10_22
        state_array = get_permuted_state(state_array, P10_22)
        state_array = get_permuted_state(state_array, P10_14)

        state_array = get_permuted_state(state_array, P)

        self.pll_state = PLL_State(state_array, position=(self.width / 2.0, self.height / 2.0), width=600)

    def sim_start(self):
        glClearColor(0.5, 0.5, 0.5, 1.0)
        pyglet.clock.schedule_interval(self.my_tick, self.sim_dt)
        pyglet.app.run()

    # Runs every frame at rate dt
    def my_tick(self, dt):
        self.render()

    def render(self):
        self.clear()
        self.pll_state.draw()


class PLL_State:
    colors = {
        "yellow": (255, 255, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "orange": (255, 100, 0)
    }
    color_names = list(colors.keys())

    batch: Batch
    tiles: ndarray
    position: Tuple[float]

    def __init__(self, tiles: np.ndarray, position: Tuple[float], width: float):
        self.position = tuple(value - width / 2 for value in position)
        self.width = width
        self.batch = Batch()
        self.tiles = tiles
        r, c = self.tiles.shape
        assert (r == c)
        self.num_rows = r
        self.rects = []

        # Compute size of components
        self.gap_portion = 0.2
        self.gap_size = self.gap_portion * width / (self.num_rows + 1)
        self.tile_portion = 1.0 - self.gap_portion
        self.tile_size = self.tile_portion * width / self.num_rows

        extremes = [0, self.num_rows - 1]

        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_rows):
                color_id = self.tiles[row_idx][col_idx]
                color = PLL_State.colors[PLL_State.color_names[color_id]]
                tile_x = self.position[0] + (col_idx + 1) * self.gap_size + col_idx * self.tile_size
                tile_y = self.position[1] + self.width - ((row_idx + 1) * (self.gap_size + self.tile_size))

                # Set batch to None for corners
                batch = None if row_idx in extremes and col_idx in extremes else self.batch
                rect = shapes.Rectangle(
                    tile_x, tile_y,
                    self.tile_size, self.tile_size,
                    color=color, batch=batch
                )
                self.rects.append(rect)
        self.last_update = time.time()

    def draw(self):
        self.batch.draw()

    @staticmethod
    def solved_PLL_state() -> ndarray:
        state_array = np.zeros((5, 5), dtype=np.int)
        state_array[0, 1:4] = PLL_State.get_color_idx("green")
        state_array[-1, 1:4] = PLL_State.get_color_idx("blue")
        state_array[1:4, 0] = PLL_State.get_color_idx("orange")
        state_array[1:4, -1] = PLL_State.get_color_idx("red")
        return state_array

    @staticmethod
    def get_color_idx(color_name: str) -> int:
        return PLL_State.color_names.index(color_name)


def create_permutation_matrix(i: int, j: int, n: int) -> ndarray:
    Pij = np.eye(n, dtype=np.int)
    temp = Pij[i, :].copy()
    Pij[i, :] = Pij[j, :]
    Pij[j, :] = temp
    return Pij


def get_permuted_state(state: ndarray, permutation: ndarray) -> ndarray:
    state_shape = state.shape
    return (permutation @ state.reshape(-1)).reshape(state_shape)


if __name__ == '__main__':
    main()
