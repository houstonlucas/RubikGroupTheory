import random
import time
from typing import Tuple, List

import numpy as np
import pyglet
from numpy import ndarray
from pyglet import shapes
from pyglet.graphics import Batch
from pyglet.gl import *

WINDOW_WIDTH = 1500


def main():
    window = SimWindow()
    window.sim_start()


class SimWindow(pyglet.window.Window):
    def __init__(self):
        super(SimWindow, self).__init__(WINDOW_WIDTH, WINDOW_WIDTH)
        self.sim_dt = 1.0 / 60.0

        state_array = PLL_State.solved_pll_state()
        n = 25

        P10_14 = create_permutation_matrix(10, 14, n)
        P10_22 = create_permutation_matrix(10, 22, n)
        P = P10_14 @ P10_22
        Pinv = P.T

        state_array2 = get_permuted_state(state_array, P)
        state_array3 = get_permuted_state(state_array, Pinv)

        pll_state = PLL_State(state_array)
        pll_state2 = PLL_State(state_array2)
        pll_state3 = PLL_State(state_array3)

        pos1 = tuple(map(float, (self.width/4.0, self.height/4.0)))
        pos2 = tuple(map(float, (self.width*3/4.0, self.height/4.0)))
        pos3 = tuple(map(float, (self.width/4.0, self.height*3/4.0)))
        state_width = self.width/4

        self.pll_state_drawer = PLL_State_Drawer()

        self.pll_state_drawer.prepare_state(pll_state, pos1, state_width)
        self.pll_state_drawer.prepare_state(pll_state2, pos2, state_width)
        self.pll_state_drawer.prepare_state(pll_state3, pos3, state_width)

    def sim_start(self):
        glClearColor(0.5, 0.5, 0.5, 1.0)
        pyglet.clock.schedule_interval(self.my_tick, self.sim_dt)
        pyglet.app.run()

    # Runs every frame at rate dt
    def my_tick(self, dt):
        self.render()

    def render(self):
        self.clear()
        self.pll_state_drawer.draw()


class PLL_State:
    tiles: ndarray
    position: Tuple[float]

    colors = {
        "yellow": (255, 255, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "orange": (255, 100, 0)
    }
    color_names = list(colors.keys())

    def __init__(self, tiles: np.ndarray):
        self.tiles = tiles

    @staticmethod
    def solved_pll_state() -> ndarray:
        state_array = np.zeros((5, 5), dtype=np.int)
        state_array[0, 1:4] = PLL_State.get_color_idx("green")
        state_array[-1, 1:4] = PLL_State.get_color_idx("blue")
        state_array[1:4, 0] = PLL_State.get_color_idx("orange")
        state_array[1:4, -1] = PLL_State.get_color_idx("red")
        return state_array

    @staticmethod
    def get_color_idx(color_name: str) -> int:
        return PLL_State.color_names.index(color_name)


class PLL_State_Drawer:
    batch: Batch
    rects: List[shapes.Rectangle]

    def __init__(self):
        self.batch = Batch()
        self.rects = []

    def prepare_state(self, pll_state: PLL_State, position: Tuple[float], width: float):

        position = tuple(value - width / 2 for value in position)

        r, c = pll_state.tiles.shape
        assert (r == c)
        num_rows = r

        # Compute size of components
        gap_portion = 0.2
        gap_size = gap_portion * width / (num_rows + 1)
        tile_portion = 1.0 - gap_portion
        tile_size = tile_portion * width / num_rows

        extremes = [0, num_rows - 1]

        for row_idx in range(num_rows):
            for col_idx in range(num_rows):
                color_id = pll_state.tiles[row_idx][col_idx]
                color = PLL_State.colors[PLL_State.color_names[color_id]]
                tile_x = position[0] + (col_idx + 1) * gap_size + col_idx * tile_size
                tile_y = position[1] + width - ((row_idx + 1) * (gap_size + tile_size))

                # Set batch to None for corners
                batch = None if row_idx in extremes and col_idx in extremes else self.batch
                rect = shapes.Rectangle(
                    tile_x, tile_y,
                    tile_size, tile_size,
                    color=color, batch=batch
                )
                self.rects.append(rect)

    def draw(self):
        self.batch.draw()


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
