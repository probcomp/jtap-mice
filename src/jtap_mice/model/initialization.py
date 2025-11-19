import jax
import genjax
import jax.numpy as jnp
from genjax import gen
from jtap_mice.distributions import uniformcat
from jtap_mice.utils import ModelOutput
from .scene_geometry import is_ball_fully_hidden, is_ball_fully_visible


@gen
def init_model(mi):
    diameter = mi.diameter
    masked_occluders = mi.masked_occluders

    x = genjax.uniform(jnp.float32(0.), mi.scene_dim[0] - diameter) @ "x"
    # for the Left-Right task, scene_dim[1] is equal to the diameter of the ball
    # hence y will always be sampled as 0
    y = genjax.uniform(jnp.float32(0.), mi.scene_dim[1] - diameter) @ "y"

    speed = genjax.uniform(jnp.float32(0.), mi.max_speed) @ "speed"
    direction = uniformcat([-1.0, 1.0]) @ "direction" # -1 is left, 1 is right

    # 0 means no collision, 1 means collision
    collision_branch = jax.lax.select(True, 0., 0.).astype(jnp.float32)
    last_collision_data = jnp.array([0,x,y,speed,direction])

    # check if the ball has hit the boundary
    hit_boundary = (x <= 0) | (x >= mi.scene_dim[0] - diameter)

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
        masked_occluders=masked_occluders, 
        is_target_hidden=is_target_hidden,
        is_target_partially_hidden=is_target_partially_hidden, 
        is_target_visible=is_target_visible,
        collision_branch=collision_branch, 
        last_collision_data=last_collision_data,
        T=jax.lax.select(True, 0, 0).astype(jnp.int32),
    )
