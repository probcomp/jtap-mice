import jax
import genjax
import jax.numpy as jnp
from genjax import gen
from jtap_mice.distributions import uniformcat
from jtap_mice.utils import ModelOutput
from .scene_geometry import is_ball_fully_hidden, is_ball_fully_visible


@gen
def init_model(mi):
    epsilon = jnp.float32(1e-5)
    diameter = mi.diameter
    masked_occluders = mi.masked_occluders

    x = genjax.uniform(-diameter, mi.scene_dim[0]) @ "x"
    # for the Left-Right task, scene_dim[1] is equal to the diameter of the ball
    # hence y will always be 0
    y = jnp.float32(0.)

    speed = genjax.uniform(jnp.float32(0.), mi.max_speed) @ "speed"
    direction = uniformcat(mi.direction_values) @ "direction" # -1 is left, 1 is right

    # check if the ball has hit the boundary
    hit_boundary = (x <= (0 + epsilon)) | (x >= (mi.scene_dim[0] - diameter - epsilon))
    is_switching_timestep = False
    is_outlier_step = False
    # check visibility condition
    is_target_hidden = is_ball_fully_hidden(x, y, diameter, masked_occluders)
    is_target_visible = is_ball_fully_visible(x, y, diameter, masked_occluders)
    is_target_partially_hidden = jnp.logical_not(jnp.logical_or(is_target_hidden, is_target_visible))


    return ModelOutput(
        diameter=diameter, 
        x=x, 
        y=y, 
        speed=speed, 
        direction=direction, 
        hit_boundary=hit_boundary,
        is_outlier_step=is_outlier_step,
        is_switching_timestep=is_switching_timestep,
        masked_occluders=masked_occluders, 
        is_target_hidden=is_target_hidden,
        is_target_partially_hidden=is_target_partially_hidden, 
        is_target_visible=is_target_visible,
        T=jax.lax.select(True, 0, 0).astype(jnp.int32),
    )
