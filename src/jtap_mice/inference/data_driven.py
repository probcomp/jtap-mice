import jax
import jax.numpy as jnp

def data_driven_size_and_position(obs, scale):
    num_obj_pixels = jnp.sum(obs == jnp.int8(2))
    obj_bool_any_axis0 = jnp.any(obs == jnp.int8(2), axis=0) # info for X axis (columns)

    max_X = obs.shape[1] * scale

    leftmost_target = scale * jnp.argmin(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), jnp.float32(jnp.inf)))
    rightmost_target = scale + scale * jnp.argmax(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), -jnp.float32(jnp.inf)))
    
    mean_diameter = rightmost_target - leftmost_target

    # Clip positions to ensure they stay within bounds
    leftmost_target = jnp.clip(leftmost_target, 0, max_X - mean_diameter)
    rightmost_target = jnp.clip(rightmost_target, 0, max_X - mean_diameter)
    
    data_driven_x = leftmost_target
    data_driven_y = jnp.float32(0.0)  # y is always 0 for this task
    is_fully_hidden = jnp.logical_not(jnp.any(obj_bool_any_axis0))

    data_driven_x = jnp.where(is_fully_hidden, jnp.float32(jnp.inf), data_driven_x)
    data_driven_y = jnp.where(is_fully_hidden, jnp.float32(jnp.inf), data_driven_y)

    return mean_diameter, data_driven_x, data_driven_y, is_fully_hidden, num_obj_pixels