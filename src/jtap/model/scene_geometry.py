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
def is_ball_in_valid_position(x, y, diameter, masked_barriers):
    return jnp.logical_not(is_ball_intersecting_rectangle(x, y, diameter, masked_barriers))
    

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

#########
# FOR EDGE AND CORNER GENERATION and Physics
#########

@jax.jit
def generate_wall_edges(scene_dim, edgemap):
    right, top = scene_dim

    wall_edges = jnp.array([
        [[jnp.float32(0), jnp.float32(0)], [jnp.float32(0), top]], # left
        [[jnp.float32(0), top], [right, top]], # top
        [[right, jnp.float32(0)], [right, top]], # right
        [[jnp.float32(0), jnp.float32(0)], [right, jnp.float32(0)]] # bottom
    ])

    edgemap["stacked_edges"] = edgemap["stacked_edges"].at[:4].set(wall_edges)
    edgemap["valid"] = edgemap["valid"].at[:4].set(jnp.ones(4, dtype=jnp.bool_))
    return edgemap

@jax.jit
def get_valid_barrier_edges(barrier):
    x, y, size_x, size_y = barrier
    left, right, top, bottom = x, x + size_x, y + size_y, y
    
    edges = jnp.array([
        [[left, bottom], [left, top]],
        [[left, top], [right, top]],
        [[right, bottom], [right, top]],
        [[left, bottom], [right, bottom]]
    ], dtype=jnp.float32)
    return edges, jnp.ones(4, dtype=jnp.bool_)

@jax.jit
def get_invalid_barrier_edges(_):
    return jnp.zeros((4, 2, 2), dtype=jnp.float32), jnp.zeros(4, dtype=jnp.bool_)

@jax.jit
def generate_barrier_edges(edgemap, masks):
    def get_barrier_edges(mask):
        return jax.lax.cond(
            mask.flag, get_valid_barrier_edges, get_invalid_barrier_edges, mask.value
        )

    stacked_edges, valid_flags = jax.vmap(get_barrier_edges)(masks)
    stacked_edges = stacked_edges.reshape(-1, 2, 2)
    valid_flags = valid_flags.flatten()

    edgemap_size = edgemap["valid"].shape[0]
    edgemap["stacked_edges"] = edgemap["stacked_edges"].at[4:edgemap_size].set(stacked_edges)
    edgemap["valid"] = edgemap["valid"].at[4:edgemap_size].set(valid_flags)
    return edgemap

@jax.jit
def get_edges_from_scene(scene_dim, edgemap, masked_barriers):
    edgemap = generate_wall_edges(scene_dim, edgemap)
    return generate_barrier_edges(edgemap, masked_barriers)

@jax.jit
def generate_target_stacked_edges(x, y, diameter):
    return jnp.array([
        # Bottom edge
        [x, y, x + diameter, y, 
                   jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)],
        # Left edge
        [x, y, x, y + diameter, 
                   jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0)],
        # Right edge
        [x + diameter, y, x + diameter, y + diameter, 
                   jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0)],
        # Top edge
        [x, y + diameter, x + diameter, y + diameter, 
                   jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0)]
    ])


def check_same_height_or_breadth_overlap(x1,x2,x3,x4, overlap = jnp.float32(0.0)):
    return jnp.logical_and(jnp.greater(x2-x3,overlap), jnp.greater(x4-x1, overlap))

def edge_intersection_time(targetedge, edgemap, vx, vy, overlap):

    no_collision_retval = jnp.array([
        jnp.float32(jnp.inf), jnp.float32(0), jnp.float32(0)]
    )

    def inner_():
        x1, y1 = targetedge[:2]
        x2, y2 = targetedge[2:4]
        bottom, left, right, top = targetedge[4:]
        x3, y3 = edgemap["stacked_edges"][0]
        x4, y4 = edgemap["stacked_edges"][1]

        # jax.debug.print(
        #     "Edge Points: (x1={x1}, y1={y1}), (x2={x2}, y2={y2}), (x3={x3}, y3={y3}), (x4={x4}, y4={y4})",
        #     x1=jnp.asarray(x1), y1=jnp.asarray(y1), 
        #     x2=jnp.asarray(x2), y2=jnp.asarray(y2), 
        #     x3=jnp.asarray(x3), y3=jnp.asarray(y3), 
        #     x4=jnp.asarray(x4), y4=jnp.asarray(y4)
        # )

        targetedge_width = x2-x1
        targetedge_height = y2-y1

        is_horizontal_case = (y1 == y2) & (y3 == y4)
        is_vertical_case = (y1 != y2) & (y3 != y4)
        
        # Handling all cases with JAX's control flow
        def horizontal_case():
            return jax.lax.cond(
                jnp.equal(vy,jnp.float32(0)),
                lambda : no_collision_retval,
                lambda : jax.lax.cond(
                    jnp.logical_and(
                        check_same_height_or_breadth_overlap(
                            jnp.round(x1 + (vx/vy)*(y3-y1), 3), 
                            jnp.round(targetedge_width + x1 + (vx/vy)*(y3-y1),3), 
                            jnp.round(x3,3), jnp.round(x4,3),
                            overlap
                        ),
                        jnp.logical_and(
                            jnp.logical_or(
                                jnp.logical_and(
                                    jnp.less(vy,jnp.float32(0)), bottom
                                ),
                                jnp.logical_and(
                                    jnp.greater(vy,jnp.float32(0)), top
                                )
                            ),
                            jnp.logical_or(
                                jnp.equal(jnp.sign(jnp.round(y3 - y1,3)),jnp.sign(jnp.round(vy,3))),
                                jnp.equal(jnp.round(y3 - y1,3), jnp.float32(0))
                            )
                        )
                    ),
                    lambda : jnp.array([jnp.round((y3 - y1)/ vy,3), jnp.float32(1), jnp.float32(0)]),
                    lambda : no_collision_retval
                )
            )

        
        def vertical_case():
            return jax.lax.cond(
                jnp.equal(vx,jnp.float32(0)),
                lambda : no_collision_retval,
                lambda : jax.lax.cond(
                    jnp.logical_and(
                        check_same_height_or_breadth_overlap(
                            jnp.round(y1 + (vy/vx)*(x3-x1),3), 
                            jnp.round(targetedge_height + y1 + (vy/vx)*(x3-x1),3), 
                            jnp.round(y3,3), jnp.round(y4,3),
                            overlap
                        ),
                        jnp.logical_and(
                            jnp.logical_or(
                                jnp.logical_and(
                                    jnp.less(vx,jnp.float32(0)), left
                                ),
                                jnp.logical_and(
                                    jnp.greater(vx,jnp.float32(0)), right
                                )
                            ),
                            jnp.logical_or( 
                                jnp.equal(jnp.sign(jnp.round(x3 - x1,3)),jnp.sign(jnp.round(vx,3))),
                                jnp.equal(jnp.round(x3 - x1,3), jnp.float32(0))
                            )
                        )
                    ),
                    lambda : jnp.array([jnp.round((x3 - x1)/ vx,3), jnp.float32(0), jnp.float32(1)]),
                    lambda : no_collision_retval
                )
            )

        return jax.lax.cond(
            is_horizontal_case,
            horizontal_case,
            lambda: jax.lax.cond(
                is_vertical_case,
                vertical_case,
                lambda: no_collision_retval
            )
        )

    return jax.lax.cond(edgemap['valid'], inner_, lambda: no_collision_retval)
                                                
    
@jax.jit
def edge_intersection_time_batched(target_edges, edgemap, vx, vy, overlap):
    """
    Fully batched version of edge intersection time computation.
    Replaces the double vmap with vectorized operations.
    
    Args:
        target_edges: (4, 8) array of target edges [x1, y1, x2, y2, bottom, left, right, top]
        edgemap: dict with 'stacked_edges' and 'valid' keys
        vx, vy: velocity components
        overlap: overlap threshold
    
    Returns:
        (4, num_edges, 3) array of [time, horizontal_flag, vertical_flag] for each target-edge pair
    """
    num_target_edges = target_edges.shape[0]
    num_scene_edges = edgemap["stacked_edges"].shape[0]
    
    # Expand dimensions for broadcasting
    # target_edges: (4, 8) -> (4, 1, 8)
    target_edges_expanded = target_edges[:, None, :]
    # scene_edges: (num_edges, 2, 2) -> (1, num_edges, 2, 2)
    scene_edges_expanded = edgemap["stacked_edges"][None, :, :, :]
    # valid: (num_edges,) -> (1, num_edges)
    valid_expanded = edgemap["valid"][None, :]
    
    # Extract coordinates
    x1 = target_edges_expanded[:, :, 0]  # (4, num_edges)
    y1 = target_edges_expanded[:, :, 1]  # (4, num_edges)
    x2 = target_edges_expanded[:, :, 2]  # (4, num_edges)
    y2 = target_edges_expanded[:, :, 3]  # (4, num_edges)
    bottom = target_edges_expanded[:, :, 4]  # (4, num_edges)
    left = target_edges_expanded[:, :, 5]    # (4, num_edges)
    right = target_edges_expanded[:, :, 6]   # (4, num_edges)
    top = target_edges_expanded[:, :, 7]     # (4, num_edges)
    
    x3 = scene_edges_expanded[:, :, 0, 0]  # (1, num_edges)
    y3 = scene_edges_expanded[:, :, 0, 1]  # (1, num_edges)
    x4 = scene_edges_expanded[:, :, 1, 0]  # (1, num_edges)
    y4 = scene_edges_expanded[:, :, 1, 1]  # (1, num_edges)
    
    # Broadcast to (4, num_edges)
    x3 = jnp.broadcast_to(x3, (num_target_edges, num_scene_edges))
    y3 = jnp.broadcast_to(y3, (num_target_edges, num_scene_edges))
    x4 = jnp.broadcast_to(x4, (num_target_edges, num_scene_edges))
    y4 = jnp.broadcast_to(y4, (num_target_edges, num_scene_edges))
    valid_broadcast = jnp.broadcast_to(valid_expanded, (num_target_edges, num_scene_edges))
    
    # Calculate edge properties
    targetedge_width = x2 - x1
    targetedge_height = y2 - y1
    
    # Determine case types
    is_horizontal_case = jnp.logical_and((y1 == y2), (y3 == y4))
    is_vertical_case = jnp.logical_and((y1 != y2), (y3 != y4))
    
    # Initialize result arrays
    no_collision_retval = jnp.array([jnp.inf, 0.0, 0.0])
    results = jnp.full((num_target_edges, num_scene_edges, 3), no_collision_retval)
    
    # Horizontal case computation
    horizontal_mask = jnp.logical_and(is_horizontal_case, valid_broadcast)
    vy_zero_mask = (vy == 0.0)
    
    # For horizontal case with non-zero vy
    horizontal_vy_nonzero = jnp.logical_and(horizontal_mask, jnp.logical_not(vy_zero_mask))
    
    # Calculate intersection parameters for horizontal case
    x_intersect = jnp.round(x1 + (vx/vy) * (y3 - y1), 3)
    x_intersect_end = jnp.round(targetedge_width + x1 + (vx/vy) * (y3 - y1), 3)
    
    # Check overlap condition
    overlap_condition = check_same_height_or_breadth_overlap(
        x_intersect, x_intersect_end, jnp.round(x3, 3), jnp.round(x4, 3), overlap
    )
    
    # Check direction conditions
    direction_condition = jnp.logical_and(
        jnp.logical_or(
            jnp.logical_and((vy < 0.0), bottom),
            jnp.logical_and((vy > 0.0), top)
        ),
        jnp.logical_or(
            (jnp.sign(jnp.round(y3 - y1, 3)) == jnp.sign(jnp.round(vy, 3))),
            (jnp.round(y3 - y1, 3) == 0.0)
        )
    )
    
    # Final horizontal condition
    horizontal_condition = jnp.logical_and(
        jnp.logical_and(horizontal_vy_nonzero, overlap_condition),
        direction_condition
    )
    
    # Calculate horizontal collision time
    horizontal_time = jnp.round((y3 - y1) / vy, 3)
    horizontal_result = jnp.stack([
        horizontal_time,
        jnp.ones_like(horizontal_time),  # horizontal flag
        jnp.zeros_like(horizontal_time)  # vertical flag
    ], axis=-1)
    
    # Update results for horizontal case
    results = jnp.where(
        horizontal_condition[..., None],
        horizontal_result,
        results
    )
    
    # Vertical case computation
    vertical_mask = jnp.logical_and(is_vertical_case, valid_broadcast)
    vx_zero_mask = (vx == 0.0)
    
    # For vertical case with non-zero vx
    vertical_vx_nonzero = jnp.logical_and(vertical_mask, jnp.logical_not(vx_zero_mask))
    
    # Calculate intersection parameters for vertical case
    y_intersect = jnp.round(y1 + (vy/vx) * (x3 - x1), 3)
    y_intersect_end = jnp.round(targetedge_height + y1 + (vy/vx) * (x3 - x1), 3)
    
    # Check overlap condition
    vertical_overlap_condition = check_same_height_or_breadth_overlap(
        y_intersect, y_intersect_end, jnp.round(y3, 3), jnp.round(y4, 3), overlap
    )
    
    # Check direction conditions
    vertical_direction_condition = jnp.logical_and(
        jnp.logical_or(
            jnp.logical_and((vx < 0.0), left),
            jnp.logical_and((vx > 0.0), right)
        ),
        jnp.logical_or(
            (jnp.sign(jnp.round(x3 - x1, 3)) == jnp.sign(jnp.round(vx, 3))),
            (jnp.round(x3 - x1, 3) == 0.0)
        )
    )
    
    # Final vertical condition
    vertical_condition = jnp.logical_and(
        jnp.logical_and(vertical_vx_nonzero, vertical_overlap_condition),
        vertical_direction_condition
    )
    
    # Calculate vertical collision time
    vertical_time = jnp.round((x3 - x1) / vx, 3)
    vertical_result = jnp.stack([
        vertical_time,
        jnp.zeros_like(vertical_time),  # horizontal flag
        jnp.ones_like(vertical_time)    # vertical flag
    ], axis=-1)
    
    # Update results for vertical case
    results = jnp.where(
        vertical_condition[..., None],
        vertical_result,
        results
    )
    
    return results

@jax.jit
def generate_wall_corners(scene_dim, cornermap):
    right, top = scene_dim

    wall_corners = jnp.array([
        [0.0, 0.0, 1, 1],    # BL
        [right, 0.0, -1, 1],  # BR
        [0.0, top, 1, -1],   # TL
        [right, top, -1, -1] # TR
    ], dtype=jnp.float32)

    cornermap["stacked_corners"] = cornermap["stacked_corners"].at[:4].set(wall_corners)
    cornermap["valid"] = cornermap["valid"].at[:4].set(jnp.ones(4, dtype=jnp.bool_))
    return cornermap

@jax.jit
def get_valid_barrier_corners(barrier):
    x, y, size_x, size_y = barrier
    left, right, top, bottom = x, x + size_x, y + size_y, y
    
    corners = jnp.array([
        [left, bottom, 1, 1],   # BL
        [right, bottom, -1, 1], # BR
        [left, top, 1, -1],    # TL
        [right, top, -1, -1]   # TR
    ], dtype=jnp.float32)
    
    return corners, jnp.ones(4, dtype=jnp.bool_)

@jax.jit
def get_invalid_barrier_corners(_):
    return jnp.zeros((4, 4), dtype=jnp.float32), jnp.zeros(4, dtype=jnp.bool_)

@jax.jit
def generate_barrier_corners(cornermap, masks):
    def get_barrier_corners(mask):
        return jax.lax.cond(
            mask.flag, get_valid_barrier_corners, get_invalid_barrier_corners, mask.value
        )

    stacked_corners, valid_flags = jax.vmap(get_barrier_corners)(masks)
    stacked_corners = stacked_corners.reshape(-1, 4)
    valid_flags = valid_flags.flatten()

    cornermap_size = cornermap["valid"].shape[0]
    cornermap["stacked_corners"] = cornermap["stacked_corners"].at[4:cornermap_size].set(stacked_corners)
    cornermap["valid"] = cornermap["valid"].at[4:cornermap_size].set(valid_flags)
    return cornermap

@jax.jit
def remove_duplicate_corners(cornermap):
    stacked_corners = cornermap["stacked_corners"]
    valid = cornermap["valid"]

    def outer_(corner_i):
        def inner_(corner_j):
            return jax.lax.cond(
                jnp.all(jnp.array([
                    valid[corner_i],
                    valid[corner_j],
                    jnp.all(stacked_corners[corner_i][:2] == stacked_corners[corner_j][:2]),
                    jnp.any(jnp.array([
                        jnp.all(stacked_corners[corner_i][2:] == -stacked_corners[corner_j][2:]),
                    ]))
                ])),
                lambda: 1,
                lambda: 0
            )

        return jnp.sum(jax.vmap(inner_)(jnp.arange(stacked_corners.shape[0])))

    duplicate_arr = jax.vmap(outer_)(jnp.arange(stacked_corners.shape[0]))
    cornermap["valid"] = jnp.where(duplicate_arr > 0, False, cornermap["valid"])
    return cornermap

@jax.jit
def get_corners_from_scene(scene_dim, cornermap, masked_barriers):
    cornermap = generate_wall_corners(scene_dim, cornermap)
    cornermap = generate_barrier_corners(cornermap, masked_barriers)
    cornermap = remove_duplicate_corners(cornermap)
    return cornermap

@jax.jit
def generate_target_stacked_corners(x, y, diameter):
    # Define the four target corners directly as JAX arrays
    corners = jnp.array([
        [x, y, 1, 1],               # Bottom-left (BL)
        [x + diameter, y, -1, 1],      # Bottom-right (BR)
        [x, y + diameter, 1, -1],      # Top-left (TL)
        [x + diameter, y + diameter, -1, -1]  # Top-right (TR)
    ], dtype=jnp.float32)
    
    return corners


@jax.jit
def reflect_velocity(vx, vy, cx, cy, x1, y1):
    # Calculate the direction vector of the line AB
    dx = x1 - cx
    dy = y1 - cy
    
    # Calculate the magnitude of the direction vector
    mag = jnp.round(jnp.sqrt(dx**2 + dy**2),3)
    
    # Normalize the direction vector to get the unit direction vector
    ux = jnp.round(dx / mag, 3)
    uy = jnp.round(dy / mag, 3)
    
    # Find the normal vector to the line AB
    nx = -uy
    ny = ux
    
    # Calculate the dot product of the velocity vector and the normal vector
    dot_product = vx * nx + vy * ny
    
    # Calculate the reflected velocity components
    vx2 = vx - jnp.float32(2) * dot_product * nx
    vy2 = vy - jnp.float32(2) * dot_product * ny
    
    return jnp.float32(-1)*vx2, jnp.float32(-1)*vy2

@jax.jit
def corner_intersection_time_circle_inner(A, B, discriminant, cx, cy, x_center, y_center, vx, vy):
    # just check x-axis coordinate to get the time if vx =/= 0, else check with vy
    t1 = jnp.round((-B + jnp.sqrt(discriminant))/(jnp.float32(2)*A),3)
    t2 = jnp.round((-B - jnp.sqrt(discriminant))/(jnp.float32(2)*A),3)

    time_taken_1 = jnp.where(jnp.logical_and(jnp.greater(t1, 0), jnp.less_equal(t1, 1)), t1, jnp.float32(jnp.inf))
    time_taken_2 = jnp.where(jnp.logical_and(jnp.greater(t2, 0), jnp.less_equal(t2, 1)), t2, jnp.float32(jnp.inf))

    min_t = jnp.min(jnp.array([time_taken_1, time_taken_2]))
    x1_new = min_t*vx + x_center
    y1_new = min_t*vy + y_center

    vx2, vy2 = reflect_velocity(vx, vy, cx, cy, x1_new, y1_new)

    return min_t, vx2, vy2

@jax.jit
def corner_intersection_time_circle(r, x, y, cornermap, vx, vy):
    # assume radius = 0.5
    # formula from https://mathworld.wolfram.com/Circle-LineIntersection.html
    # https://chatgpt.com/share/66fb6b4d-9134-8004-848a-f1f2c319394f
    def inner():
        cx, cy = cornermap["stacked_corners"][0], cornermap["stacked_corners"][1]
        x1, y1, x2, y2 = x - cx + r,y - cy + r, x + vx - cx + r, y + vy - cy + r # assume corner centered at 0, 0
        dx = x2 - x1
        dy = y2 - y1
        A = jnp.square(dx) + jnp.square(dy)
        B = jnp.float32(2)*(x1*dx + y1*dy)
        C = jnp.square(x1) + jnp.square(y1) - jnp.square(r)
        discriminant = jnp.round(jnp.square(B) - jnp.float32(4)*A*C,3)
        # NOTE: positive discriminant means that velocity
        # vector intersects, does not mean that it is in the correct direction

        # handling discrete jumps
        gap_2 = jnp.square(x1) + jnp.square(y1)
        # peturb object in velocity (less than radial distance) to see if it is moving towards barrier
        coefs = jnp.arange(0.1,1.01,0.01)
        speed = jnp.sqrt(jnp.square(vx) + jnp.square(vy))
        dts = coefs *jnp.min(jnp.array([r,speed]))/speed
        gaps_after_2 = jnp.square(x1 + dts*vx) + jnp.square(y1+dts*vy)
        moving_towards_barrier = jnp.all(jnp.less(jnp.round(gaps_after_2,3), jnp.round(gap_2, 3)))
        penetration = jnp.less(jnp.round(gap_2, 3), jnp.round(jnp.square(r),3))

        return jax.lax.cond(
            penetration,
            lambda: jax.lax.cond(
                moving_towards_barrier,
                lambda: (jnp.float32(0.), *reflect_velocity(vx, vy, cx, cy, x+r, y+r)),
                lambda: (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf))
            ),
            lambda: jax.lax.cond(
                jnp.logical_and(
                    jnp.greater(discriminant, jnp.float32(0)),
                    moving_towards_barrier
                ),
                corner_intersection_time_circle_inner,
                lambda *_ : (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf)),
                A, B, discriminant, cx, cy, x+r, y+r, vx, vy
            )
        )

    return jax.lax.cond(cornermap["valid"], inner, lambda: (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf)))

@jax.jit
def corner_intersection_time_circle_batched(x, y, diameter, cornermap, vx, vy):
    """
    Fully batched version of corner intersection time computation.
    Replaces the vmap with vectorized operations.
    
    Args:
        x, y: ball position
        diameter: ball diameter
        cornermap: dict with 'stacked_corners' and 'valid' keys
        vx, vy: velocity components
    
    Returns:
        tuple of (min_time, vx2, vy2, all_times)
    """
    radius = jnp.float32(0.5) * diameter
    num_corners = cornermap["stacked_corners"].shape[0]
    
    # Extract corner data
    corners = cornermap["stacked_corners"]  # (num_corners, 4)
    valid = cornermap["valid"]  # (num_corners,)
    
    # Extract corner coordinates and normals
    cx = corners[:, 0]  # (num_corners,)
    cy = corners[:, 1]  # (num_corners,)
    nx = corners[:, 2]  # (num_corners,)
    ny = corners[:, 3]  # (num_corners,)
    
    # Transform coordinates relative to corner
    x1 = x - cx + radius  # (num_corners,)
    y1 = y - cy + radius  # (num_corners,)
    x2 = x + vx - cx + radius  # (num_corners,)
    y2 = y + vy - cy + radius  # (num_corners,)
    
    # Calculate line-circle intersection parameters
    dx = x2 - x1  # (num_corners,)
    dy = y2 - y1  # (num_corners,)
    A = dx**2 + dy**2  # (num_corners,)
    B = 2 * (x1 * dx + y1 * dy)  # (num_corners,)
    C = x1**2 + y1**2 - radius**2  # (num_corners,)
    discriminant = jnp.round(B**2 - 4 * A * C, 3)  # (num_corners,)
    
    # Check for penetration (ball already inside corner)
    gap_2 = x1**2 + y1**2  # (num_corners,)
    penetration = jnp.less(jnp.round(gap_2, 3), jnp.round(radius**2, 3))  # (num_corners,)
    
    # Check if moving towards barrier
    coefs = jnp.arange(0.1, 1.01, 0.01)  # (91,)
    speed = jnp.sqrt(vx**2 + vy**2)
    dts = coefs * jnp.min(jnp.array([radius, speed])) / speed  # (91,)
    
    # Broadcast for vectorized computation
    x1_broadcast = x1[:, None]  # (num_corners, 1)
    y1_broadcast = y1[:, None]  # (num_corners, 1)
    dts_broadcast = dts[None, :]  # (1, 91)
    vx_broadcast = vx
    vy_broadcast = vy
    
    gaps_after = (x1_broadcast + dts_broadcast * vx_broadcast)**2 + (y1_broadcast + dts_broadcast * vy_broadcast)**2  # (num_corners, 91)
    gap_2_broadcast = gap_2[:, None]  # (num_corners, 1)
    
    moving_towards_barrier = jnp.all(jnp.less(jnp.round(gaps_after, 3), jnp.round(gap_2_broadcast, 3)), axis=1)  # (num_corners,)
    
    # Initialize result arrays
    corner_times = jnp.full(num_corners, jnp.inf)
    vx2s = jnp.full(num_corners, jnp.inf)
    vy2s = jnp.full(num_corners, jnp.inf)
    
    # Handle penetration case
    penetration_mask = penetration & valid & moving_towards_barrier
    
    # Vectorized reflection calculation for penetration case
    # Calculate direction vector from corner to ball center
    dx_pen = (x + radius) - cx  # (num_corners,)
    dy_pen = (y + radius) - cy  # (num_corners,)
    
    # Calculate magnitude and normalize
    mag_pen = jnp.sqrt(dx_pen**2 + dy_pen**2)
    ux_pen = dx_pen / mag_pen
    uy_pen = dy_pen / mag_pen
    
    # Find normal vector
    nx_pen = -uy_pen
    ny_pen = ux_pen
    
    # Calculate dot product and reflection
    dot_product_pen = vx * nx_pen + vy * ny_pen
    vx2_pen_all = -(vx - 2 * dot_product_pen * nx_pen)
    vy2_pen_all = -(vy - 2 * dot_product_pen * ny_pen)
    
    corner_times = jnp.where(penetration_mask, 0.0, corner_times)
    vx2s = jnp.where(penetration_mask, vx2_pen_all, vx2s)
    vy2s = jnp.where(penetration_mask, vy2_pen_all, vy2s)
    
    # Handle intersection case
    intersection_mask = (~penetration) & valid & (discriminant > 0) & moving_towards_barrier
    
    # Calculate intersection times
    sqrt_discriminant = jnp.sqrt(discriminant)
    t1 = jnp.round((-B + sqrt_discriminant) / (2 * A), 3)
    t2 = jnp.round((-B - sqrt_discriminant) / (2 * A), 3)
    
    # Filter valid times (0 < t <= 1)
    valid_t1 = (t1 > 0) & (t1 <= 1)
    valid_t2 = (t2 > 0) & (t2 <= 1)
    
    # Choose minimum valid time
    time_taken_1 = jnp.where(valid_t1, t1, jnp.inf)
    time_taken_2 = jnp.where(valid_t2, t2, jnp.inf)
    min_t = jnp.min(jnp.stack([time_taken_1, time_taken_2]), axis=0)
    
    # Calculate new position and reflected velocity (vectorized)
    x1_new = min_t * vx + x + radius  # (num_corners,)
    y1_new = min_t * vy + y + radius  # (num_corners,)
    
    # Vectorized reflection calculation for intersection case
    dx_int = x1_new - cx  # (num_corners,)
    dy_int = y1_new - cy  # (num_corners,)
    
    # Calculate magnitude and normalize
    mag_int = jnp.sqrt(dx_int**2 + dy_int**2)
    ux_int = dx_int / mag_int
    uy_int = dy_int / mag_int
    
    # Find normal vector
    nx_int = -uy_int
    ny_int = ux_int
    
    # Calculate dot product and reflection
    dot_product_int = vx * nx_int + vy * ny_int
    vx2_int = -(vx - 2 * dot_product_int * nx_int)
    vy2_int = -(vy - 2 * dot_product_int * ny_int)
    
    # Update results for intersection case
    corner_times = jnp.where(intersection_mask, min_t, corner_times)
    vx2s = jnp.where(intersection_mask, vx2_int, vx2s)
    vy2s = jnp.where(intersection_mask, vy2_int, vy2s)
    
    # Find minimum time and corresponding velocities
    min_idx = corner_times.argmin()
    min_time = corner_times[min_idx]
    min_vx2 = vx2s[min_idx]
    min_vy2 = vy2s[min_idx]
    
    return min_time, min_vx2, min_vy2, corner_times