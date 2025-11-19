import jax.numpy as jnp
import genjax
from genjax import gen
from jtap_mice.distributions import direction_flip_distribution
from jtap_mice.utils import ModelOutput
from .scene_geometry import is_ball_fully_hidden, is_ball_fully_visible

@gen
def stepper_model(mo, mi, inference_mode_bool):

    x = mo.x
    y = mo.y
    speed = mo.speed
    direction = mo.direction
    diameter = mo.diameter
    model_direction_flip_prob = mi.model_direction_flip_prob
    pos_noise = mi.σ_pos
    speed_noise = mi.σ_speed
    masked_occluders = mi.masked_occluders

    scene_length = mi.scene_dim[0]

    
    # this one line is the "physics" of the model
    x_mean = x + (speed * direction)

    # we have to make sure that the ball does not go out of bounds. If it does, then we record it in the state and clip it to the boundary
    epsilon = jnp.float32(1e-5)
    x_mean = jnp.clip(x_mean, epsilon, scene_length - diameter - epsilon)

    # sample an outlier probability
    is_outlier_step = genjax.flip(mi.model_outlier_prob) @ "is_outlier_step"

    # if this is an outlier step, sample position and speed from extremely high-variance truncated normal distributions and the direction flip at probability of 0.5.

    pos_noise = jnp.where(is_outlier_step, jnp.float32(1e5), pos_noise)
    speed_noise = jnp.where(is_outlier_step, jnp.float32(1e5), speed_noise)
    model_direction_flip_prob = jnp.where(is_outlier_step, jnp.float32(0.5), model_direction_flip_prob)
    
    new_x = genjax.truncated_normal(x_mean, pos_noise, jnp.float32(0.), scene_length - diameter) @ 'x'
    new_y = y
    new_speed = genjax.truncated_normal(speed, speed_noise, jnp.float32(0.), mi.max_speed) @ "speed"
    new_direction = direction_flip_distribution(direction, model_direction_flip_prob) @ "direction"

    hit_boundary = (new_x <= (epsilon)) | (new_x >= (scene_length - diameter - epsilon))
    is_switching_timestep = jnp.not_equal(new_direction, direction)

    # check visibility condition
    is_target_hidden = is_ball_fully_hidden(new_x, new_y, diameter, masked_occluders)
    is_target_visible = is_ball_fully_visible(new_x, new_y, diameter, masked_occluders)
    is_target_partially_hidden = jnp.logical_not(jnp.logical_or(is_target_hidden, is_target_visible))

    return ModelOutput(
        diameter=diameter,
        x=new_x,
        y=new_y,
        speed=new_speed,
        direction=new_direction,
        hit_boundary=hit_boundary,
        is_switching_timestep=is_switching_timestep,
        masked_occluders=mo.masked_occluders,
        is_target_hidden=is_target_hidden,
        is_target_partially_hidden=is_target_partially_hidden,
        is_target_visible=is_target_visible,
        T=mo.T + jnp.int32(1)
    )