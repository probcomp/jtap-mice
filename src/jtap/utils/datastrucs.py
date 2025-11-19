import chex
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
import dataclasses
from typing import Tuple, List, Dict
from genjax import Mask

from jtap.model.scene_geometry import get_edges_from_scene, get_corners_from_scene
from jtap.utils.stimuli import JTAPStimulus
from jtap.utils.common_math import d2r, r2d

@chex.dataclass(frozen=True)
class ChexModelInput:
    model_outlier_prob: float
    proposal_direction_outlier_tau: float
    proposal_direction_outlier_alpha: float
    σ_pos: float
    σ_speed: float
    σ_NOCOL_direction: float
    σ_COL_direction: float
    σ_speed_init_model: float
    σ_direction_init_model: float
    pixel_corruption_prob: float
    tile_size: int
    σ_pixel_spatial: float
    image_power_beta: float
    max_speed: float
    max_num_barriers: float
    max_num_occ: float
    num_x_grid : int
    num_y_grid : int
    grid_size_bounds : Tuple[float, float]
    max_num_col_iters : int
    image_discretization: float = None
    diameter: float = None
    scene_dim: Tuple[int, int] = None
    num_x_grid_arr : jnp.ndarray = None
    num_y_grid_arr : jnp.ndarray = None
    pix_x: jnp.ndarray = None
    pix_y: jnp.ndarray = None
    tile_size_arr: jnp.ndarray = None
    masked_barriers : Mask = None
    masked_occluders : Mask = None
    edgemap : Dict = None
    cornermap : Dict = None
    red_sensor : jnp.ndarray = None
    green_sensor : jnp.ndarray = None
    simulate_every : int = None
    σ_pos_sim: float = None
    σ_speed_sim: float = None
    σ_NOCOL_direction_sim: float = None
    σ_COL_direction_sim: float = None
    σ_speed_occ: float = None
    σ_NOCOL_direction_occ: float = None
    σ_COL_direction_occ: float = None
    σ_pos_initprop: float = None
    σ_speed_initprop: float = None
    σ_NOCOL_direction_initprop: float = None
    σ_NOCOL_direction_stepprop: float = None
    σ_COL_direction_prop : float = None
    σ_speed_stepprop: float = None
    σ_pos_stepprop: float = None

    def prepare_hyperparameters(self):
        # NOTE: Diameter is a constant value of 1.0 for now
        self.update("num_x_grid_arr", jnp.zeros(self.num_x_grid))
        self.update("num_y_grid_arr", jnp.zeros(self.num_y_grid))
        self.update("tile_size_arr", jnp.zeros(self.tile_size))
        if self.diameter is None:
            # NOTE: Diameter is a constant value of 1.0 if not specified
            self.update("diameter", jnp.float32(1.0))
        # depend on inference params
        if self.σ_pos_sim is None:
            self.update("σ_pos_sim", self.σ_pos)
        if self.σ_speed_sim is None:
            self.update("σ_speed_sim", self.σ_speed)
        # depend on simulation params
        # depend on simulation params
        if self.σ_NOCOL_direction_sim is None:
            self.update("σ_NOCOL_direction_sim", self.σ_NOCOL_direction)
        if self.σ_COL_direction_sim is None:
            self.update("σ_COL_direction_sim", self.σ_COL_direction)
        # depend on sim_params
        if self.σ_speed_occ is None:
            self.update("σ_speed_occ", self.σ_speed_sim)
        
        if self.σ_NOCOL_direction_occ is None:
            self.update("σ_NOCOL_direction_occ", self.σ_NOCOL_direction_sim)
        if self.σ_COL_direction_occ is None:
            self.update("σ_COL_direction_occ", self.σ_COL_direction_sim)
        # depend on simulation params
        if self.σ_pos_initprop is None:
            self.update("σ_pos_initprop", self.σ_pos_sim)
        if self.σ_speed_initprop is None:
            self.update("σ_speed_initprop", self.σ_speed_sim)
        if self.σ_NOCOL_direction_initprop is None:
            self.update("σ_NOCOL_direction_initprop", self.σ_NOCOL_direction_sim)
        # depend on initprops
        if self.σ_NOCOL_direction_stepprop is None:
            self.update("σ_NOCOL_direction_stepprop", self.σ_NOCOL_direction_initprop)
        if self.σ_COL_direction_prop is None:
            self.update("σ_COL_direction_prop", self.σ_COL_direction_sim)
        if self.σ_speed_stepprop is None:
            self.update("σ_speed_stepprop", self.σ_speed_initprop)
        if self.σ_pos_stepprop is None:
            self.update("σ_pos_stepprop", self.σ_pos_initprop)

    def update(self, attr, value):
        object.__setattr__(self, attr, value)

    def prepare_scene_geometry(self, stimulus):


        assert isinstance(stimulus, JTAPStimulus)

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
        masked_barriers, masked_occluders = extract_masked_barriers_and_occluders(self.image_discretization, self.max_num_barriers, self.max_num_occ, initial_discrete_obs)
        self.update('masked_barriers', masked_barriers)
        self.update('masked_occluders', masked_occluders)
        max_num_edges = 4 + self.max_num_barriers * 4
        max_num_corners = 4 + self.max_num_barriers * 4
        empty_edgemap = {   
            "stacked_edges": jnp.zeros((max_num_edges, 2, 2), dtype=jnp.float32),
            "valid": jnp.zeros(max_num_edges, dtype=jnp.bool_),
        }
        empty_cornermap = {
            "stacked_corners": jnp.zeros((max_num_corners, 4), dtype=jnp.float32),
            "valid": jnp.zeros(max_num_corners, dtype=jnp.bool_),
        }
        self.update('edgemap', get_edges_from_scene(self.scene_dim, empty_edgemap, masked_barriers))
        self.update('cornermap', get_corners_from_scene(self.scene_dim, empty_cornermap, masked_barriers))

        red_sensor_rect, _ = extract_disjoint_rectangles(initial_discrete_obs == 4, scale = self.image_discretization)
        green_sensor_rect, _ = extract_disjoint_rectangles(initial_discrete_obs == 5, scale = self.image_discretization)
        self.update('red_sensor', jnp.array(red_sensor_rect[0]))
        self.update('green_sensor', jnp.array(green_sensor_rect[0]))
        
# default mi template
default_mi = ChexModelInput(
    model_outlier_prob=jnp.float32(0.0),
    proposal_direction_outlier_tau=d2r(jnp.float32(200.0)),
    proposal_direction_outlier_alpha=jnp.float32(50.0),
    σ_pos=jnp.float32(0.0005),
    σ_speed=jnp.float32(0.1),
    σ_NOCOL_direction=jnp.float32(175.0),
    σ_COL_direction=jnp.float32(100.0),
    σ_speed_init_model=jnp.float32(0.5),
    σ_direction_init_model=jnp.float32(360.0),
    pixel_corruption_prob=jnp.float32(0.47),
    tile_size=jnp.int32(3),
    σ_pixel_spatial=jnp.float32(10.0),
    image_power_beta=jnp.float32(0.1),
    max_speed=jnp.float32(1.0),
    max_num_barriers=jnp.int32(10),
    max_num_occ=jnp.int32(5),
    image_discretization=jnp.float32(0.1),
    num_x_grid=jnp.int32(11),
    num_y_grid=jnp.int32(11),
    grid_size_bounds=jnp.array([0.2, 1.5]),
    max_num_col_iters = jnp.float32(10),
    simulate_every = jnp.int32(1)
)

ModelOutput = namedtuple("ModelOutput", [
    "diameter", "x", "y", "speed", "direction",
    "masked_occluders", "masked_barriers", "is_target_hidden",
    "is_target_partially_hidden", "is_target_visible",
    "red_sensor", "green_sensor", "collision_branch", "last_collision_data",
    "edgemap", "cornermap", "T", "stopped_early"
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

def extract_masked_barriers_and_occluders(image_discretization, max_num_barriers, max_num_occ, initial_discrete_obs):
    def make_masked_stack(vals, check):
        return Mask(flag = check, value = tuple(vals))
    make_masked_stack_vmap = jax.vmap(make_masked_stack)

    empty_rectangle = jnp.array([0.0, 0.0, 0.0, 0.0], dtype = jnp.float32)

    scale = image_discretization
    barrier_rects, num_barriers = extract_disjoint_rectangles(initial_discrete_obs == 3, scale = scale)
    occ_rects, num_occ = extract_disjoint_rectangles(initial_discrete_obs == 1, scale = scale)

    fixed_barrier_rects = jnp.array([
        barrier_rects[i] if i < num_barriers else empty_rectangle
        for i in range(max_num_barriers)
    ])
    fixed_occ_rects = jnp.array([
        occ_rects[i] if i < num_occ else empty_rectangle
        for i in range(max_num_occ)
    ])
    masked_barriers = make_masked_stack_vmap(fixed_barrier_rects, jnp.arange(max_num_barriers) < num_barriers)
    masked_occluders = make_masked_stack_vmap(fixed_occ_rects, jnp.arange(max_num_occ) < num_occ)
    return masked_barriers, masked_occluders

