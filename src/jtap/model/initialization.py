import jax
import genjax
import jax.numpy as jnp
from genjax import gen
from jtap.distributions import uniformposition2d
from jtap.utils import ModelOutput
from .scene_geometry import get_edges_from_scene, get_corners_from_scene, is_ball_fully_hidden, is_ball_fully_visible


@gen
def init_model(mi):
    diameter = mi.diameter
    masked_barriers = mi.masked_barriers
    masked_occluders = mi.masked_occluders
    edgemap = mi.edgemap
    cornermap = mi.cornermap

    x, y = uniformposition2d(jnp.float32(0.), mi.scene_dim[0] - diameter, jnp.float32(0.), mi.scene_dim[1] - diameter) @ "xy" # joint sampling of x and y
    speed = genjax.uniform(jnp.float32(0.), mi.max_speed) @ "speed"
    direction = genjax.uniform(-jnp.pi, jnp.pi) @ "direction"

    # bug workaround for not being able to return non-traced values for importance
    collision_branch = jax.lax.select(True, 4., 4.).astype(jnp.float32)
    # timestep, x, y, speed, direction
    last_collision_data = jnp.array([0,x,y,speed,direction])

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
        masked_occluders=masked_occluders, 
        masked_barriers=masked_barriers, 
        is_target_hidden=is_target_hidden,
        is_target_partially_hidden=is_target_partially_hidden, 
        is_target_visible=is_target_visible, 
        red_sensor=mi.red_sensor, 
        green_sensor=mi.green_sensor, 
        collision_branch=collision_branch, 
        last_collision_data=last_collision_data,
        edgemap=edgemap, 
        cornermap=cornermap, 
        T=jax.lax.select(True, 0, 0).astype(jnp.int32),
        stopped_early=jax.lax.select(False, False, False).astype(jnp.bool)
    )
