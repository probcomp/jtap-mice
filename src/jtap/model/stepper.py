import jax.numpy as jnp
import genjax
from genjax import gen
from jtap.distributions import circular_normal
from .collision import velocity_transform
from jtap.utils import ModelOutput

@gen
def stepper_model(mo, mi, inference_mode_bool):

    diameter = mo.diameter
    speed = mo.speed
    # Different hyperparameters for inference vs simulation
    pos_noise = jnp.where(inference_mode_bool, mi.σ_pos, mi.σ_pos_sim)

    speed_noise, next_direction, direction_σ, next_x, next_y, \
    next_vx, next_vy, collision_branch, is_target_hidden, is_target_visible, \
        is_target_partially_hidden, stopped_early = velocity_transform(mo, mi, diameter, inference_mode_bool)

    # different speed and proposal on first timestep
    first_timestep_inference_mode_bool = jnp.equal(mo.T, jnp.int32(0)) & inference_mode_bool
    direction_noise = jnp.where(first_timestep_inference_mode_bool, mi.σ_direction_init_model, direction_σ)
    speed_noise = jnp.where(first_timestep_inference_mode_bool, mi.σ_speed_init_model, speed_noise)

    # sample an outlier probability
    is_outlier_step = genjax.flip(mi.model_outlier_prob) @ "is_outlier_step"

    # if this is an outlier step, sample direction from a massive circular normal (direction_scale = 1e5) not during simulation mode
    direction_noise = jnp.where(jnp.logical_and(inference_mode_bool, is_outlier_step), jnp.float32(1e5), direction_noise)

    new_speed = genjax.truncated_normal(speed, speed_noise, jnp.float32(0.), mi.max_speed) @ "speed"
    new_direction = circular_normal(next_direction, direction_noise) @ "direction"

    # create dependency in model by editing next_x and next_y based on the sampled speed and direction
    sampled_vx, sampled_vy = jnp.cos(new_direction) * new_speed, jnp.sin(new_direction) * new_speed
    delta_vx, delta_vy = sampled_vx - next_vx, sampled_vy - next_vy
    next_x, next_y = next_x + delta_vx, next_y + delta_vy


    new_x = genjax.truncated_normal(next_x, pos_noise, jnp.float32(0.), mi.scene_dim[0] - diameter) @ 'x'
    new_y = genjax.truncated_normal(next_y, pos_noise, jnp.float32(0.), mi.scene_dim[1] - diameter) @ 'y'

    # mo in previous timestep was fully hidden and now it is partially hidden (or fully visible)
    # this means it is the first timestep after reappearance
    reappearance_tstep = jnp.logical_and(mo.is_target_hidden, jnp.logical_or(is_target_partially_hidden, is_target_visible))

    retain_collision_data = jnp.logical_and(jnp.logical_not(reappearance_tstep), jnp.equal(collision_branch, jnp.float32(4)))

    last_collision_data = jnp.where(
        retain_collision_data,
        mo.last_collision_data,
        jnp.array([mo.T + jnp.int32(1), new_x, new_y, new_speed, new_direction])
    )
    return ModelOutput(
        diameter=diameter, x=new_x, y=new_y, speed=new_speed, direction=new_direction, 
        masked_occluders=mo.masked_occluders, 
        masked_barriers=mo.masked_barriers, is_target_hidden=is_target_hidden, 
        is_target_partially_hidden=is_target_partially_hidden, is_target_visible=is_target_visible, 
        red_sensor=mo.red_sensor, green_sensor=mo.green_sensor, 
        collision_branch=collision_branch, last_collision_data=last_collision_data, 
        edgemap=mo.edgemap, cornermap=mo.cornermap, T=mo.T + jnp.int32(1),
        stopped_early=stopped_early
    )

@gen
def stepper_model_no_obs(mo, mi, inference_mode_bool):

    diameter = mo.diameter
    speed = mo.speed
    # Different hyperparameters for inference vs simulation

    speed_noise, next_direction, direction_σ, next_x, next_y, \
    next_vx, next_vy, collision_branch, is_target_hidden, is_target_visible, \
        is_target_partially_hidden, stopped_early = velocity_transform(mo, mi, diameter, inference_mode_bool)

    # different speed and proposal on first timestep
    first_timestep_inference_mode_bool = jnp.equal(mo.T, jnp.int32(0)) & inference_mode_bool
    direction_noise = jnp.where(first_timestep_inference_mode_bool, mi.σ_direction_init_model, direction_σ)
    speed_noise = jnp.where(first_timestep_inference_mode_bool, mi.σ_speed_init_model, speed_noise)

    # sample an outlier probability
    is_outlier_step = genjax.flip(mi.model_outlier_prob) @ "is_outlier_step"

    # if this is an outlier step, sample direction from a massive circular normal (direction_scale = 1e5) not during simulation mode
    direction_noise = jnp.where(jnp.logical_and(inference_mode_bool, is_outlier_step), jnp.float32(1e5), direction_noise)

    new_speed = genjax.truncated_normal(speed, speed_noise, jnp.float32(0.), mi.max_speed) @ "speed"
    new_direction = circular_normal(next_direction, direction_noise) @ "direction"

    # create dependency in model by editing next_x and next_y based on the sampled speed and direction
    sampled_vx, sampled_vy = jnp.cos(new_direction) * new_speed, jnp.sin(new_direction) * new_speed
    delta_vx, delta_vy = sampled_vx - next_vx, sampled_vy - next_vy
    next_x, next_y = next_x + delta_vx, next_y + delta_vy


    new_x = next_x
    new_y = next_y

    # mo in previous timestep was fully hidden and now it is partially hidden (or fully visible)
    # this means it is the first timestep after reappearance
    reappearance_tstep = jnp.logical_and(mo.is_target_hidden, jnp.logical_or(is_target_partially_hidden, is_target_visible))

    retain_collision_data = jnp.logical_and(jnp.logical_not(reappearance_tstep), jnp.equal(collision_branch, jnp.float32(4)))

    last_collision_data = jnp.where(
        retain_collision_data,
        mo.last_collision_data,
        jnp.array([mo.T + jnp.int32(1), new_x, new_y, new_speed, new_direction])
    )
    return ModelOutput(
        diameter=diameter, x=new_x, y=new_y, speed=new_speed, direction=new_direction, 
        masked_occluders=mo.masked_occluders, 
        masked_barriers=mo.masked_barriers, is_target_hidden=is_target_hidden, 
        is_target_partially_hidden=is_target_partially_hidden, is_target_visible=is_target_visible, 
        red_sensor=mo.red_sensor, green_sensor=mo.green_sensor, 
        collision_branch=collision_branch, last_collision_data=last_collision_data, 
        edgemap=mo.edgemap, cornermap=mo.cornermap, T=mo.T + jnp.int32(1),
        stopped_early=stopped_early
    )