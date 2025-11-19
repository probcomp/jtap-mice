import jax
import genjax
from genjax import gen, ChoiceMapBuilder as C
import jax.numpy as jnp
from typing import NamedTuple
from jtap_mice.model import likelihood_model, is_ball_in_scene_track

class GridData(NamedTuple):
    grid_proposed_xs: jnp.ndarray
    sampled_idxs: jnp.ndarray
    sampled_x_cells: jnp.ndarray
    grid_size: jnp.ndarray
    x_grid_center: jnp.ndarray
    x_grid: jnp.ndarray
    grid_logprobs: jnp.ndarray
    num_obj_pixels: jnp.ndarray

def find_valid_positions_bool(pos_chm, diameter, scene_length):
    # y is always 0
    return jax.vmap(is_ball_in_scene_track, in_axes=(0, None, None))(pos_chm['x'], diameter, scene_length)

grid_likelihood_evaluator = jax.vmap(
    likelihood_model.logpdf,
    in_axes=(
        None,  # obs_image
        (None, None, None, 0, 0, None, None),  # render_args -- see get_render_args in model.py (pix_x, pix_y, diameter, x, y, masked_occluders, image_discretization)
        None,  # pixel_corruption_prob
        None,  # tile_size_arr
        None,  # σ_pixel_spatial
        None   # image_power_beta
    )
)

def adaptive_grid_size(num_obj_pixels, grid_size_bounds):
    """
    Adaptive grid size based on the number of object pixels. inversely proportional to the number of object pixels.
    Args:
        num_obj_pixels: Number of object pixels
        grid_size_bounds: Tuple of minimum and maximum grid size
    Returns:
        Grid size
    """
    proportion_obj_pixels = num_obj_pixels / 80
    grid_size = grid_size_bounds[1] - ((grid_size_bounds[1] - grid_size_bounds[0]) * proportion_obj_pixels)
    return grid_size

def make_position_grid(mi, x_center, x_size):
    # 1D grid: only over x dimension, y always 0.
    num_x_grid = mi.num_x_grid_arr.shape[0]
    min_x, max_x = x_center - x_size/2, x_center + x_size/2
    # Clip to scene bounds
    min_x = jnp.clip(min_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0))
    max_x = jnp.clip(max_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0))
    # Handle inf center
    min_x = jnp.where(jnp.isinf(x_center), mi.scene_dim[0] + x_size/2, min_x)
    max_x = jnp.where(jnp.isinf(x_center), mi.scene_dim[0] + 3*x_size/2, max_x)
    # Set up grid
    x_grid = jnp.linspace(min_x, max_x, num_x_grid)
    # position_grid is only over x, y is always 0
    position_grid = jax.vmap(lambda x: C['x'].set(x).at['y'].set(jnp.float32(0.)))(x_grid)
    # Cells (intervals)
    if num_x_grid > 1:
        x_div = x_grid[1] - x_grid[0]
    else:
        x_div = jnp.float32(1.0)
    pos_cells = jnp.stack([x_grid - x_div/2, x_grid + x_div/2], axis=1)
    return position_grid, pos_cells

@gen
def sample_grid(grid_scores, pos_cells):
    sampled_grid_idx = genjax.categorical(grid_scores) @ 'sampled_grid_index'
    sampled_x_cell = pos_cells[sampled_grid_idx, 0:2]
    sampled_x = genjax.uniform(sampled_x_cell[0], sampled_x_cell[1]) @ 'x'
    return sampled_x, sampled_grid_idx, sampled_x_cell

grid_proposer = jax.vmap(sample_grid.propose, in_axes=(0,None))