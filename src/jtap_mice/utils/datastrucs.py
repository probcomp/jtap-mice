import chex
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
import dataclasses
from typing import Tuple, List, Dict
from genjax import Mask

from jtap_mice.utils.stimuli import JTAPMiceStimulus
from jtap_mice.utils.common_math import d2r, r2d

@chex.dataclass(frozen=True)
class ChexModelInput:
    model_outlier_prob: float
    proposal_direction_outlier_tau: float
    proposal_direction_outlier_alpha: float
    σ_pos: float
    σ_speed: float
    model_direction_flip_prob: float
    pixel_corruption_prob: float
    tile_size: int
    σ_pixel_spatial: float
    image_power_beta: float
    max_speed: float
    max_num_occ: int
    num_x_grid : int
    num_y_grid : int
    grid_size_bounds : Tuple[float, float]
    direction_values: List[float] = None
    image_discretization: float = None
    diameter: float = None
    scene_dim: Tuple[int, int] = None
    num_x_grid_arr : jnp.ndarray = None
    pix_x: jnp.ndarray = None
    pix_y: jnp.ndarray = None
    tile_size_arr: jnp.ndarray = None
    masked_occluders : Mask = None
    simulate_every : int = None
    σ_pos_sim: float = None
    σ_speed_sim: float = None
    σ_direction_sim: float = None
    σ_pos_initprop: float = None
    σ_direction_stepprop: float = None
    σ_speed_stepprop: float = None
    σ_pos_stepprop: float = None

    def prepare_hyperparameters(self):
        # NOTE: Diameter is a constant value of 1.0 for now
        self.update("num_x_grid_arr", jnp.zeros(self.num_x_grid))
        self.update("tile_size_arr", jnp.zeros(self.tile_size))

    def update(self, attr, value):
        object.__setattr__(self, attr, value)

    def prepare_scene_geometry(self, stimulus):

        assert isinstance(stimulus, JTAPMiceStimulus)

        self.update("image_discretization", jnp.float32(1/stimulus.pixel_density))

        initial_discrete_obs = stimulus.discrete_obs[0]
        # NOTE: scene_dim is flipped to match the coordinate system of the stimulus (i,j) convert to--> (x,y)
        scene_dim = jnp.array([self.image_discretization * initial_discrete_obs.shape[1], self.image_discretization * initial_discrete_obs.shape[0]])
        self.update('scene_dim', scene_dim)
        diameter = stimulus.diameter
        self.update('diameter', jnp.float32(diameter))
        self.update("pix_x", jnp.array(np.arange(0, self.scene_dim[0], self.image_discretization)).astype(jnp.float32))
        # NOTE: pix_y is flipped to match the coordinate system of the stimulus
        self.update("pix_y", jnp.array(np.arange(0, self.scene_dim[1], self.image_discretization)[::-1]).astype(jnp.float32))
        masked_occluders = extract_masked_occluders(self.image_discretization, self.max_num_occ, initial_discrete_obs)
        self.update('masked_occluders', masked_occluders)
        self.update('direction_values', jnp.array([-1.0, 1.0], dtype=jnp.float32))


ModelOutput = namedtuple("ModelOutput", [
    "diameter", "x", "y", "speed", "direction", "hit_boundary", "is_switching_timestep", "masked_occluders", "is_target_hidden", "is_target_partially_hidden",
    "is_target_visible", "T"
])

def extract_disjoint_rectangles(mask, scale = 0.1):
    rectangles = []
    visited = np.zeros_like(mask, dtype=bool)
    labeled_mask = np.zeros_like(mask, dtype=int)  # Array to hold labeled rectangles
    n, m = mask.shape
    label = 1  # Start labeling rectangles from 1

    def find_rectangle(x, y):
        """Find the width and height of the rectangle starting at (x, y)."""
        # Determine width of the rectangle by expanding horizontally
        width = 0
        while y + width < m and mask[x, y + width] and not visited[x, y + width]:
            width += 1

        # Determine height by expanding vertically
        height = 0
        while x + height < n and all(mask[x + height, y:y + width]) and not any(visited[x + height, y:y + width]):
            height += 1

        # Mark the found rectangle as visited and label it
        for i in range(x, x + height):
            for j in range(y, y + width):
                visited[i, j] = True
                labeled_mask[i, j] = label

        return width, height

    # Iterate over each cell in the mask
    for i in range(n):
        for j in range(m):
            if mask[i, j] and not visited[i, j]:
                width, height = find_rectangle(i, j)
                rectangles.append((j * scale, scale * (n - i - height), scale*width, scale*height))
                label += 1  # Increment label for the next rectangle

    return rectangles, len(rectangles)

def extract_masked_occluders(image_discretization, max_num_occ, initial_discrete_obs):
    def make_masked_stack(vals, check):
        return Mask(flag=check, value=tuple(vals))
    make_masked_stack_vmap = jax.vmap(make_masked_stack)

    empty_rectangle = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    scale = image_discretization
    occ_rects, num_occ = extract_disjoint_rectangles(initial_discrete_obs == 1, scale=scale)

    fixed_occ_rects = jnp.array([
        occ_rects[i] if i < num_occ else empty_rectangle
        for i in range(max_num_occ)
    ])
    masked_occluders = make_masked_stack_vmap(fixed_occ_rects, jnp.arange(max_num_occ) < num_occ)
    return masked_occluders

