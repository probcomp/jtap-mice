import jax
import jax.numpy as jnp

def data_driven_size_and_position(obs, scale):
    num_obj_pixels = jnp.sum(obs == jnp.int8(2))
    obj_bool_any_axis0 = jnp.any(obs == jnp.int8(2), axis=0) # collapsing all rows to extract info for X axis (columns)
    obj_bool_any_axis1 = jnp.any(obs == jnp.int8(2), axis=1) # collapsing all columns to extract info for Y axis (rows)

    max_Y = obs.shape[0] * scale
    max_X = obs.shape[1] * scale

    topmost_target = max_Y - (scale * jnp.argmin(jnp.where(obj_bool_any_axis1, jnp.arange(obs.shape[0]), jnp.float32(jnp.inf))))
    bottommost_target = max_Y - (scale + scale * jnp.argmax(jnp.where(obj_bool_any_axis1, jnp.arange(obs.shape[0]), -jnp.float32(jnp.inf))))
    leftmost_target = scale * jnp.argmin(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), jnp.float32(jnp.inf)))
    rightmost_target = scale + scale * jnp.argmax(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), -jnp.float32(jnp.inf)))
    
    mean_diameter = 0.5 * (rightmost_target - leftmost_target) + 0.5 * (topmost_target - bottommost_target)
    
    # Clip positions to ensure they stay within bounds
    bottommost_target = jnp.clip(bottommost_target, 0, max_Y - mean_diameter)
    topmost_target = jnp.clip(topmost_target, 0, max_Y - mean_diameter)
    leftmost_target = jnp.clip(leftmost_target, 0, max_X - mean_diameter)
    rightmost_target = jnp.clip(rightmost_target, 0, max_X - mean_diameter)
    
    data_driven_x, data_driven_y = leftmost_target, bottommost_target
    is_fully_hidden = jnp.logical_not(jnp.logical_or(jnp.any(obj_bool_any_axis0), jnp.any(obj_bool_any_axis1)))

    data_driven_x = jnp.where(is_fully_hidden, jnp.float32(jnp.inf), data_driven_x)
    data_driven_y = jnp.where(is_fully_hidden, jnp.float32(jnp.inf), data_driven_y)

    return mean_diameter, data_driven_x, data_driven_y, is_fully_hidden, num_obj_pixels