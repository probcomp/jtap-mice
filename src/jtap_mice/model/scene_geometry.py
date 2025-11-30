import jax
import jax.numpy as jnp

#########
# FOR SCENE QUERIES
#########

def is_circle_intersecting_box(box_x, box_y, box_w, box_h, circle_x, circle_y, radius):
    closest_x = jnp.clip(circle_x, box_x, box_x + box_w)
    closest_y = jnp.clip(circle_y, box_y, box_y + box_h)
    
    dist_sq = (closest_x - circle_x) ** 2 + (closest_y - circle_y) ** 2
    
    return dist_sq <= radius ** 2

def is_ball_intersecting_rectangle_inner(masked_rectangle, x, y, diameter):
    occ_x, occ_y, occ_size_x, occ_size_y = masked_rectangle.value
    r = diameter/2
    return jnp.logical_and(
        masked_rectangle.flag,
        is_circle_intersecting_box(occ_x, occ_y, occ_size_x, occ_size_y, x+r, y+r, r)
    )

@jax.jit
def is_ball_intersecting_rectangle(x, y, diameter, masked_rectangles):
    """
    Use to check if the ball is in an invalid position or if it is partiall hidden
    """
    return jnp.any(jax.vmap(is_ball_intersecting_rectangle_inner, in_axes = (0,None,None,None))(masked_rectangles, x, y, diameter))

@jax.jit
def is_ball_fully_visible(x, y, diameter, masked_occluders):
    return jnp.logical_not(is_ball_intersecting_rectangle(x, y, diameter, masked_occluders))

@jax.jit
def is_ball_in_scene_track(x, diameter, scene_length, epsilon = 1e-5):
    return jnp.logical_and(x >= -diameter + epsilon, x <= scene_length - diameter - epsilon)

@jax.jit
def is_ball_fully_hidden_inner(masked_occluder, x, y, diameter):
    occ_x, occ_y, occ_size_x, occ_size_y = masked_occluder.value
    return jnp.logical_and(jnp.all(
        jnp.array([
            x + diameter <= occ_x + occ_size_x, 
            x >= occ_x,
            y + diameter <= occ_y + occ_size_y, 
            y >= occ_y
        ])
    ), masked_occluder.flag)

@jax.jit
def is_ball_fully_hidden(x, y, diameter, masked_occluders):
    return jnp.any(jax.vmap(is_ball_fully_hidden_inner, in_axes = (0,None,None,None))(masked_occluders, x, y, diameter))