from __future__ import annotations
import random
import time
from typing import Tuple, List, Union

import numpy as np
import pyglet
from numpy import ndarray
from pyglet import shapes
from pyglet.graphics import Batch
from pyglet.gl import *


def main():
    window_width = 1500
    window_height = 500
    viewer = PllViewer(window_width, window_height)

    p10_14 = create_PLL_permutation(10, 14)
    p10_22 = create_PLL_permutation(10, 22)
    edge_cycle = p10_14 @ p10_22
    Pinv = edge_cycle.T

    pll_state_drawer = PllStateDrawer()
    generated_state_list = generate_state_list(edge_cycle, 3)
    pll_state_drawer.prepare_pll_list(generated_state_list, (viewer.width, viewer.height))

    viewer.add_drawer(pll_state_drawer)

    viewer.start()


class PllViewer(pyglet.window.Window):
    sim_dt: float
    drawers: List[PllStateDrawer]

    def __init__(self, width: int, height: int):
        super(PllViewer, self).__init__(width, height)
        self.sim_dt = 1.0 / 60.0
        self.drawers = []

    def start(self):
        grey_shade = 0.25
        glClearColor(grey_shade, grey_shade, grey_shade, 1.0)
        pyglet.clock.schedule_interval(self.my_tick, self.sim_dt)
        pyglet.app.run()

    # Runs every frame at rate dt
    def my_tick(self, dt):
        self.render()

    def render(self):
        self.clear()
        glPushMatrix()
        glTranslated(0, self.height, 0)
        glScaled(1, -1, 1)
        for drawer in self.drawers:
            drawer.draw()
        glPopMatrix()

    def add_drawer(self, drawer: PllStateDrawer):
        self.drawers.append(drawer)


class PllState:
    tiles: ndarray
    permutation: ndarray

    colors = {
        "yellow": (255, 255, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "orange": (255, 100, 0)
    }
    color_names = list(colors.keys())

    def __init__(self, tiles: ndarray = None, permutation: ndarray = None):
        self.tiles = tiles
        self.permutation = permutation
        if self.tiles is None:
            self.tiles = PllState.solved_pll_tiles()
        if self.permutation is None:
            self.permutation = np.eye(25)

    def __mul__(self, other) -> PllState:
        # self * other
        rhs_permutation = None
        if type(other) == self.__class__:
            rhs_permutation = other.permutation
        elif type(other) == ndarray:
            rhs_permutation = other
        assert (rhs_permutation is not None)

        resultant_permutation = self.permutation @ rhs_permutation
        resultant_tiles = get_permuted_tiles(self.tiles, rhs_permutation)
        resultant_state = PllState(resultant_tiles, resultant_permutation)
        return resultant_state

    @staticmethod
    def solved_pll_tiles() -> ndarray:
        state_array = np.zeros((5, 5), dtype=np.int)
        state_array[0, 1:4] = PllState.get_color_idx("green")
        state_array[-1, 1:4] = PllState.get_color_idx("blue")
        state_array[1:4, 0] = PllState.get_color_idx("orange")
        state_array[1:4, -1] = PllState.get_color_idx("red")
        return state_array

    @staticmethod
    def get_color_idx(color_name: str) -> int:
        return PllState.color_names.index(color_name)


class PllStateDrawer:
    batch: Batch
    rects: List[shapes.Rectangle]

    def __init__(self, batch: Batch = None):
        if batch is None:
            self.batch = Batch()
        self.rects = []

    def prepare_state(self, pll_state: PllState, state_position: Tuple[float], width: float):

        state_position = tuple(value - width / 2 for value in state_position)

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
                color = PllState.colors[PllState.color_names[color_id]]
                tile_x = state_position[0] + (col_idx + 1) * gap_size + col_idx * tile_size
                tile_y = state_position[1] + ((row_idx + 1) * (gap_size + tile_size))

                # Set batch to None for corners
                batch = None if row_idx in extremes and col_idx in extremes else self.batch
                rect = shapes.Rectangle(
                    tile_x, tile_y,
                    tile_size, tile_size,
                    color=color, batch=batch
                )
                self.rects.append(rect)

    def prepare_pll_list(self, pll_states: List[PllState], as_row: bool = True):
        state_width = self.state_size
        positions = self.__get_pll_list_positions(pll_states, as_row)

        for state_idx, state in enumerate(pll_states):
            self.prepare_state(state, positions[state_idx], state_width)

    def __get_pll_list_positions(self, pll_states: List[PllState], row_vector: bool = True) -> List[Tuple[float]]:
        positions = []
        for state_idx, state in enumerate(pll_states):
            x_i = self.state_size * (state_idx + 0.5)
            y_i = self.state_size / 2.0

            if row_vector:
                position = tuple(map(float, (x_i, y_i)))
            else:
                position = tuple(map(float, (y_i, x_i)))

            positions.append(position)
        return positions

    def draw(self):
        self.batch.draw()


def generate_state_list(permutation: ndarray, list_size: int) -> List[PllState]:
    state_list = [PllState()]
    for state_idx in range(1, list_size):
        state = state_list[-1] * permutation
        state_list.append(state)

    return state_list


def create_PLL_permutation(i: int, j: int) -> ndarray:
    return create_permutation_matrix(i, j, 25)


def create_permutation_matrix(i: int, j: int, n: int) -> ndarray:
    Pij = np.eye(n, dtype=np.int)
    temp = Pij[i, :].copy()
    Pij[i, :] = Pij[j, :]
    Pij[j, :] = temp
    return Pij


def get_permuted_tiles(state: ndarray, permutation: ndarray) -> ndarray:
    state_shape = state.shape
    return (state.reshape(-1) @ permutation).reshape(state_shape)


if __name__ == '__main__':
    main()
