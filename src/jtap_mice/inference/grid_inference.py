import jax
import genjax
from genjax import gen, ChoiceMapBuilder as C
import jax.numpy as jnp
from typing import NamedTuple
from jtap_mice.model import likelihood_model, is_ball_in_valid_position

class GridData(NamedTuple):
    grid_proposed_xs: jnp.ndarray
    grid_proposed_ys: jnp.ndarray
    sampled_idxs: jnp.ndarray
    sampled_x_cells: jnp.ndarray
    sampled_y_cells: jnp.ndarray
    grid_size: jnp.ndarray
    x_grid_center: jnp.ndarray
    y_grid_center: jnp.ndarray
    x_grid: jnp.ndarray
    y_grid: jnp.ndarray
    grid_logprobs: jnp.ndarray
    num_obj_pixels: jnp.ndarray

def find_valid_positions_bool(pos_chm, diameter, masked_barriers):
    return jax.vmap(is_ball_in_valid_position, in_axes=(0,0,None,None))(pos_chm['x'], pos_chm['y'], diameter, masked_barriers)

grid_likelihood_evaluator = jax.vmap(
    likelihood_model.logpdf,
    in_axes = (None,(
        None,None,None,0,0,None,None,None,None,None
    ),None, None, None, None)
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


def make_position_grid(mi, x_center, y_center, x_size, y_size):
    # make grid with num_x by num_y
    num_x_grid = mi.num_x_grid_arr.shape[0]
    num_y_grid = mi.num_y_grid_arr.shape[0]
    min_x, max_x = x_center - x_size/2, x_center + x_size/2
    min_y, max_y = y_center - y_size/2, y_center + y_size/2

    # Clip to scene bounds
    min_x, max_x = jnp.clip(min_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0)), jnp.clip(max_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0))
    min_y, max_y = jnp.clip(min_y, jnp.float32(0.0), mi.scene_dim[1] - jnp.float32(1.0)), jnp.clip(max_y, jnp.float32(0.0), mi.scene_dim[1] - jnp.float32(1.0))


    # if x_center and y_center are jnp.inf, then set min and max outside the scene bounds
    # the extra 3*x_size/2 and 3*y_size/2 is to ensure the grid is outside the scene bounds
    min_x = jnp.where(jnp.isinf(x_center), mi.scene_dim[0] + x_size/2, min_x)
    max_x = jnp.where(jnp.isinf(x_center), mi.scene_dim[0] + 3*x_size/2, max_x)
    min_y = jnp.where(jnp.isinf(y_center), mi.scene_dim[1] + y_size/2, min_y)
    max_y = jnp.where(jnp.isinf(y_center), mi.scene_dim[1] + 3*y_size/2, max_y)

    # create deterministic grid points
    x_grid = jnp.linspace(min_x, max_x, num_x_grid)
    y_grid = jnp.linspace(min_y, max_y, num_y_grid)
    
    # make meshgrid and choicemap grid
    x_meshgrid, y_meshgrid = jnp.meshgrid(x_grid, y_grid, indexing = 'ij')
    x_meshgrid_flat, y_meshgrid_flat = x_meshgrid.flatten(), y_meshgrid.flatten()
    position_grid = jax.vmap(lambda x,y: C['x'].set(x).at['y'].set(y))(x_meshgrid_flat, y_meshgrid_flat)
    # make cells/intervals and modify tail bounds
    x_div = x_grid[2] - x_grid[1]
    y_div = y_grid[2] - y_grid[1]
    pos_cells = jnp.stack([x_meshgrid_flat - x_div/2, x_meshgrid_flat + x_div/2, y_meshgrid_flat - y_div/2, y_meshgrid_flat + y_div/2], axis = 1)

    return position_grid, pos_cells

@gen
def sample_grid(grid_scores, pos_cells):
    sampled_grid_idx = genjax.categorical(grid_scores) @ 'sampled_grid_index'
    sampled_x_cell = pos_cells[sampled_grid_idx, 0:2]
    sampled_y_cell = pos_cells[sampled_grid_idx, 2:4]
    sampled_x = genjax.uniform(sampled_x_cell[0], sampled_x_cell[1]) @ 'x'
    sampled_y = genjax.uniform(sampled_y_cell[0], sampled_y_cell[1]) @ 'y'
    return sampled_x,sampled_y,sampled_grid_idx,sampled_x_cell,sampled_y_cell

grid_proposer = jax.vmap(sample_grid.propose, in_axes=(0,None))