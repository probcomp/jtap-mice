import jax.numpy as jnp
import jax
from .initialization import is_ball_fully_hidden, is_ball_fully_visible
from .scene_geometry import edge_intersection_time_batched, corner_intersection_time_circle_batched, generate_target_stacked_edges
from jax.debug import print as jprint

def maybe_resolve_collision(diameter, x,y, vx, vy, edgemap, cornermap, dist_to_travel):

    speed = jnp.sqrt(vx**2 + vy**2) # can be different from dist_to_travel
    time_left = jnp.round(dist_to_travel/speed,3)

    overlap = 0.50 - 1e-3
    edge_collision_data = edge_intersection_time_batched(
        generate_target_stacked_edges(x,y, diameter), edgemap, vx, vy, overlap
    )

    t_edges = edge_collision_data[...,0].flatten()
    t_edges_horizontal = jnp.where(edge_collision_data[...,1].flatten(), t_edges, jnp.float32(jnp.inf))
    t_edges_vertical = jnp.where(edge_collision_data[...,2].flatten(), t_edges, jnp.float32(jnp.inf))

    # return edges
    min_edge_time_horizontal = t_edges_horizontal.min()
    edge_time_horizontal = jnp.where(
        min_edge_time_horizontal <= time_left,
        min_edge_time_horizontal, jnp.float32(jnp.inf)
    )
    min_edge_time_vertical = t_edges_vertical.min()
    edge_time_vertical = jnp.where(
        min_edge_time_vertical <= time_left,
        min_edge_time_vertical, jnp.float32(jnp.inf)
    )


    min_corner_time, vx2, vy2, c_times = corner_intersection_time_circle_batched(
        x, y, diameter, cornermap, vx, vy,
    )

    corner_time = jnp.where(
        min_corner_time <= time_left,
        min_corner_time, jnp.float32(jnp.inf)
    )

    result = jnp.select(
        [
            jnp.isfinite(edge_time_horizontal) & (edge_time_horizontal == edge_time_vertical) & (edge_time_horizontal <= corner_time),
            jnp.isfinite(edge_time_horizontal) & (edge_time_horizontal < edge_time_vertical) & (edge_time_horizontal <= corner_time),
            jnp.isfinite(edge_time_vertical) & (edge_time_vertical <= corner_time),
            jnp.isfinite(corner_time)
        ],
        jnp.array([
            [
                1,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - edge_time_horizontal * speed, 3),  # remaining_dist
                x + vx * edge_time_horizontal,  # new_x
                y + vy * edge_time_horizontal,  # new_y
                vx * -1,  # new_vx
                vy * -1,  # new_vy
                0.,  # collision_branch
            ],
            # Horizontal edge collision
            [
                0,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - edge_time_horizontal * speed, 3),  # remaining_dist
                x + vx * edge_time_horizontal,  # new_x
                y + vy * edge_time_horizontal,  # new_y
                vx * 1,  # new_vx
                vy * -1,  # new_vy
                1.,  # collision_branch
            ],
            # Vertical edge collision
            [
                1,  # x_collision
                0,  # y_collision
                jnp.round(dist_to_travel - edge_time_vertical * speed, 3),  # remaining_dist
                x + vx * edge_time_vertical,  # new_x
                y + vy * edge_time_vertical,  # new_y
                vx * -1,  # new_vx
                vy * 1,  # new_vy
                2.,  # collision_branch
            ],
            # Corner collision
            [
                1,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - corner_time * speed, 3),  # remaining_dist
                x + vx * corner_time,  # new_x
                y + vy * corner_time,  # new_y
                vx2,  # new_vx
                vy2,  # new_vy
                3.,  # collision_branch
            ]
        ], dtype= jnp.float32),
        jnp.array([
            0,  # x_collision
            0,  # y_collision
            0.,  # remaining_dist
            x + vx * time_left,  # new_x
            y + vy * time_left,  # new_y
            vx,  # new_vx
            vy,  # new_vy
            4.,  # collision_branch
        ], dtype= jnp.float32)
    )

    return result, c_times

def velocity_transform(mo, mi, diameter, inference_mode_bool):
    speed, direction = mo.speed, mo.direction
    vx, vy = speed * jnp.cos(direction), speed * jnp.sin(direction)
    dist_to_travel = jnp.sqrt(vx**2 + vy**2)
    next_x, next_y, next_vx, next_vy = mo.x, mo.y, vx, vy
    edgemap = mo.edgemap
    cornermap = mo.cornermap

    def loop_condition(carry):
        *_, dist_to_travel, col_iter = carry
        physics_cond = jnp.logical_and(
            jnp.logical_not(jnp.isclose(dist_to_travel, jnp.float32(0), atol=1e-03)), 
            jnp.greater_equal(dist_to_travel, jnp.float32(0))
        )
        early_stop_cond = jnp.less(col_iter, mi.max_num_col_iters)
        return jnp.logical_and(physics_cond, early_stop_cond)

    def loop_body(carry):
        diameter, collision_detect_x, collision_detect_y, old_x, old_y, old_vx, old_vy, collision_branch, edgemap, cornermap, dist_to_travel, col_iter = carry
        (collision_detect_x, collision_detect_y, dist_to_travel, 
         next_x, next_y, next_vx, next_vy, new_collision_branch), c_times = maybe_resolve_collision(
            diameter, old_x, old_y, old_vx, old_vy, edgemap, cornermap, dist_to_travel
        )
        collision_branch = jnp.where(
            jnp.equal(new_collision_branch, jnp.float32(4.)), 
            collision_branch, 
            new_collision_branch
        )
        return (
            diameter, collision_detect_x, collision_detect_y, next_x, next_y, 
            next_vx, next_vy, collision_branch, edgemap, cornermap, 
            dist_to_travel, col_iter + jnp.float32(1)
        )

    initial_carry = (
        diameter, jnp.float32(0), jnp.float32(0), next_x, next_y, 
        next_vx, next_vy, jnp.float32(4), edgemap, cornermap, 
        dist_to_travel, jnp.float32(0)
    )
    
    _, collision_detect_x, collision_detect_y, next_x, next_y, next_vx, next_vy, collision_branch, *_, col_iter = jax.lax.while_loop(
        loop_condition, loop_body, initial_carry
    )

    stopped_early = jnp.greater_equal(col_iter, mi.max_num_col_iters)

    # Visibility checks
    is_target_hidden = is_ball_fully_hidden(next_x, next_y, diameter, mo.masked_occluders)
    is_target_visible = is_ball_fully_visible(next_x, next_y, diameter, mo.masked_occluders)
    is_target_partially_hidden = jnp.logical_not(jnp.logical_or(is_target_hidden, is_target_visible))

    # Different hyperparameters for inference vs simulation
    speed_noise = jnp.where(inference_mode_bool, mi.σ_speed, mi.σ_speed_sim)
    σ_COL_direction = jnp.where(inference_mode_bool, mi.σ_COL_direction, mi.σ_COL_direction_sim)
    σ_NOCOL_direction = jnp.where(inference_mode_bool, mi.σ_NOCOL_direction, mi.σ_NOCOL_direction_sim)

    direction_σ = jnp.where(collision_branch == jnp.float32(4), σ_NOCOL_direction, σ_COL_direction)
    next_direction = jnp.arctan2(next_vy, next_vx)

    return (
        speed_noise, next_direction, direction_σ, next_x, next_y, next_vx, next_vy,
        collision_branch, is_target_hidden, is_target_visible,
        is_target_partially_hidden, stopped_early
    )