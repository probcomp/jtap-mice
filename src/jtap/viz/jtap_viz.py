import math
import os

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from IPython.display import HTML as HTML_Display
from jax.scipy.special import logsumexp
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Arrow, Circle
from PIL import Image
from scipy.stats import vonmises
from tqdm import tqdm

from jtap.utils import discrete_obs_to_rgb, slice_pt
from jtap.inference import JTAPData
from jtap.evaluation import compute_weight_component_correlations, get_rg_raw_beliefs

def rerun_jtap_single_run(
    JTAP_DATA,
    rgb_video_highres = None,
    stimulus_name = 'jtap_single_runv0',
    override_prediction_length = None,
    tracking_dot_size_range = (1,5),
    prediction_line_size_range = (1,5),
    jtap_run_idx = None,
    render_grid = False,
    grid_dot_radius = 1.0,
    show_velocity = False  # ADDED: show velocity arrows if True
):
    assert isinstance(JTAP_DATA, JTAPData)
    
    # Handle multiple runs
    if JTAP_DATA.num_jtap_runs > 1:
        if jtap_run_idx is None:
            print(f"Multiple JTAP runs detected ({JTAP_DATA.num_jtap_runs} runs). No index provided, visualizing index 0 (first run).")
            jtap_run_idx = 0
        else:
            if jtap_run_idx >= JTAP_DATA.num_jtap_runs:
                raise ValueError(f"jtap_run_idx {jtap_run_idx} is out of range. Available runs: 0 to {JTAP_DATA.num_jtap_runs - 1}")
    else:
        jtap_run_idx = 0

    # Local scoped imports
    from scipy.special import logsumexp

    # extract discrete obs
    discrete_obs = JTAP_DATA.stimulus.discrete_obs
    rgb_video = discrete_obs_to_rgb(discrete_obs)

    ###########################
    # PRE-PROCESS JTAP DATA
    ###########################

    # Helper function to get indexed data
    def get_run_data(data, run_idx=None):
        """Extract data for a specific run if multiple runs exist"""
        if JTAP_DATA.num_jtap_runs > 1:
            return slice_pt(data, run_idx)
        return data

    # Step 0: Get basic params
    image_height = rgb_video.shape[1]
    image_width = rgb_video.shape[2]
    inference_input = JTAP_DATA.params.inference_input
    pixel_density = round(1/get_run_data(inference_input.image_discretization, jtap_run_idx))
    diameter = get_run_data(inference_input.diameter, jtap_run_idx)
    max_speed = get_run_data(inference_input.max_speed, jtap_run_idx)
    ESS_over_time = get_run_data(JTAP_DATA.inference.ESS, jtap_run_idx)
    ESS_threshold = get_run_data(JTAP_DATA.params.ESS_threshold, jtap_run_idx)
    num_particles = get_run_data(JTAP_DATA.params.num_particles, jtap_run_idx)
    resampled_over_time = get_run_data(JTAP_DATA.inference.resampled, jtap_run_idx)
    simulate_every = get_run_data(JTAP_DATA.params.simulate_every, jtap_run_idx)

    # Step 1: Compute inference and prediction lengths
    num_inference_steps = get_run_data(JTAP_DATA.params.max_inference_steps, jtap_run_idx)
    num_prediction_steps = get_run_data(JTAP_DATA.params.max_prediction_steps, jtap_run_idx)
    num_prediction_steps = num_prediction_steps if override_prediction_length is None else override_prediction_length

    # Step 2: Extract inference/tracking data
    weights = get_run_data(JTAP_DATA.inference.weight_data.final_weights, jtap_run_idx)  # T by N_PARTICLES
    tracking_x = get_run_data(JTAP_DATA.inference.tracking.x, jtap_run_idx)  # T by N_PARTICLES
    tracking_y = get_run_data(JTAP_DATA.inference.tracking.y, jtap_run_idx)  # T by N_PARTICLES
    tracking_direction = get_run_data(JTAP_DATA.inference.tracking.direction, jtap_run_idx)  # T by N_PARTICLES
    tracking_speed = get_run_data(JTAP_DATA.inference.tracking.speed, jtap_run_idx)  # T by N_PARTICLES

    # Step 2.5: Create prediction initialization arrays that freeze values every simulate_every timesteps
    # This mimics how predictions are initialized from fixed reference states
    block_indices = (np.arange(num_inference_steps) // simulate_every) * simulate_every  # T
    pred_init_x = tracking_x[block_indices]  # T by N_PARTICLES
    pred_init_y = tracking_y[block_indices]  # T by N_PARTICLES

    # Step 3: Extract prediction data, and concat with tracking data
    prediction_x = get_run_data(JTAP_DATA.inference.prediction.x, jtap_run_idx)  # T by num_prediction_steps by N_PARTICLES
    prediction_y = get_run_data(JTAP_DATA.inference.prediction.y, jtap_run_idx)  # T by num_prediction_steps by N_PARTICLES
    prediction_x = jnp.concatenate([pred_init_x[:,None,:], prediction_x], axis = 1)  # T by num_prediction_steps+1 by N_PARTICLES
    prediction_y = jnp.concatenate([pred_init_y[:,None,:], prediction_y], axis = 1)  # T by num_prediction_steps+1 by N_PARTICLES

    # Step 4: Compute normalized weights over time (T by N_PARTICLES)
    normalized_nonlog_weights = np.exp(weights - logsumexp(weights, axis = 1, keepdims = True)) # T by N_PARTICLES
    # normalize weights to sum to 1, while weights are mathematically normalized,
    # this is to get rid of errors in the weights due to numerical precision
    normalized_nonlog_weights /= normalized_nonlog_weights.sum(axis = 1, keepdims = True)
    max_nonlog_weight = np.max(normalized_nonlog_weights, axis = 1) # find max nonlog weight at each timestep (Shape: T)

    # step 5: Compute tracking dot sizes accoring to normalized weights
    min_dot_radii, max_dot_radii = (sz/2 for sz in tracking_dot_size_range)
    tracking_dot_radii = ((normalized_nonlog_weights/max_nonlog_weight[:,None]) * (max_dot_radii - min_dot_radii)) + min_dot_radii # T by N_PARTICLES
    

    # step 6: Compute prediction line sizes accoring to normalized weights
    min_line_radii, max_line_radii = (sz/2 for sz in prediction_line_size_range)
    prediction_line_radii = ((normalized_nonlog_weights/max_nonlog_weight[:,None]) * (max_line_radii - min_line_radii)) + min_line_radii # T by num_prediction_steps+1 by N_PARTICLES

    # --- Also create an array for particle velocity arrow thickness, same shape as tracking_dot_radii ---
    velocity_arrow_radii = ((normalized_nonlog_weights/max_nonlog_weight[:,None]) * (max_line_radii - min_line_radii)) + min_line_radii  # T by N_PARTICLES

    # step 7: define a function to convert x and y in point space to rerun image UV space
    xy_to_uv = lambda x, y: ((x + 0.5*diameter) * pixel_density, image_height - ((y + 0.5*diameter) * pixel_density))

    # step 8: convert tracking and prediction data to rerun image UV space
    tracking_u, tracking_v = xy_to_uv(tracking_x, tracking_y) # T by N_PARTICLES each
    tracking_uv = np.stack([tracking_u, tracking_v], axis = 2) # T by N_PARTICLES by 2
    prediction_u, prediction_v = xy_to_uv(prediction_x, prediction_y) # T by num_prediction_steps+1 by N_PARTICLES each

    # NOTE: that the N_particles dimension is lifted while the num predictions step dimension is brought down
    prediction_uv = np.stack([prediction_u, prediction_v], axis = 3).swapaxes(1, 2) # NOTE: T by N_PARTICLES by num_prediction_steps+1 by 2

    # Step 9: Compute rg expectation. Output is T by 3 (green, red, uncertain)
    rg_raw_beliefs = get_run_data(get_rg_raw_beliefs(JTAP_DATA, num_prediction_steps), jtap_run_idx)

    
    # Step 10: Compute speed and direction samples
    n_samples = 1000

    num_speed_bins = int(max_speed/ 0.1) + 1
    rng = np.random.default_rng()
    sampled_speeds = np.array([
        rng.choice(tracking_speed[t], size=n_samples, p=normalized_nonlog_weights[t])
        for t in range(num_inference_steps)
    ])  # T by n_samples

    # Compute speed histogram bins for all timesteps at once (fully batched)
    speed_bin_edges = np.linspace(0, max_speed, num_speed_bins + 1)
    # Use numpy's histogram function with 2D input to batch process all timesteps
    speed_bin_counts = np.array([
        np.histogram(sampled_speeds[t], bins=speed_bin_edges, density=True)[0]
        for t in range(num_inference_steps)
    ])  # T by num_speed_bins
    speed_bin_counts /= speed_bin_counts.max()
    speed_bin_centers = (speed_bin_edges[:-1] + speed_bin_edges[1:]) / 2

    num_direction_bins = 90
    sampled_directions = np.array([
        rng.choice(tracking_direction[t], size=n_samples, p=normalized_nonlog_weights[t])
        for t in range(num_inference_steps)
    ])  # T by n_samples

    # Compute direction histogram bins for all timesteps at once (fully batched)
    direction_bin_edges = np.linspace(-np.pi, np.pi, num_direction_bins + 1)
    # Use numpy's histogram function with 2D input to batch process all timesteps
    direction_bin_counts = np.array([
        np.histogram(sampled_directions[t], bins=direction_bin_edges, density=True)[0]
        for t in range(num_inference_steps)
    ])  # T by num_direction_bins
    direction_bin_counts /= direction_bin_counts.max()
    direction_bin_centers = (direction_bin_edges[:-1] + direction_bin_edges[1:]) / 2

    # Step 11: Compute weight component correlations
    weight_component_correlations = compute_weight_component_correlations(JTAP_DATA, run_idx=jtap_run_idx)

    ################################################################################
    # Process grid data for rendering if requested
    ################################################################################
    grid_positions_per_frame = None
    if render_grid and hasattr(JTAP_DATA.inference, "grid_data"):
        # Get the grid for the first particle (they are the same for every particle)
        grid_x = get_run_data(JTAP_DATA.inference.grid_data.x_grid, jtap_run_idx)  # shape: [T, num_grid_cells]
        grid_y = get_run_data(JTAP_DATA.inference.grid_data.y_grid, jtap_run_idx)
        if grid_x is not None and grid_y is not None:
            # Prepare: for each timestep, select valid grid points inside scene bounds, transform for rerun rendering.
            grid_positions_per_frame = []
            has_valid_grid_over_time = []
            for t in range(grid_x.shape[0]):
                # NOTE: No Grid for first timestep
                if t == 0:
                    positions = np.zeros((0,2))
                    grid_positions_per_frame.append(positions)
                    has_valid_grid_over_time.append(False)
                    continue
                x_t = grid_x[t]   # shape (num_grid_cells,)
                y_t = grid_y[t]
                # Consider cells inside the scene (x in [0, image_width/pixel_density - diameter], y in [0, image_height/pixel_density - diameter])
                # We'll check scene bounds in image units (scene coordinates)
                # Since coordinate system is 0-centered and diameter in scene units, usually grid values are from -diameter/2 to scene-diameter/2
                left = -0.5*diameter
                right = (image_width/pixel_density) - 0.5*diameter
                top = -0.5*diameter
                bottom = (image_height/pixel_density) - 0.5*diameter
                in_bounds = (
                    (x_t >= left) & (x_t <= right) &
                    (y_t >= top) & (y_t <= bottom)
                )
                grid_x_in = x_t[in_bounds]
                grid_y_in = y_t[in_bounds]
                if np.sum(in_bounds) > 0:
                    u, v = xy_to_uv(grid_x_in, grid_y_in)
                    # Stack as Nx2 array for rerun Points2D
                    positions = np.stack([u, v], axis=-1)
                    has_valid_grid_over_time.append(True)
                else:
                    positions = np.zeros((0,2))  # No valid grid points for this frame
                    has_valid_grid_over_time.append(False)
                grid_positions_per_frame.append(positions)

    #####################
    # LOG DATA TO RERUN
    #####################

    rerun_app_id = "jtap_single_run" if stimulus_name is None else stimulus_name
    rr.init(rerun_app_id, spawn=False)
    rr.connect_grpc()
    rr.log("/", rr.Clear(recursive=True))

    # primitive datatypes
    black_color = [0,0,0,255]
    yellow_color = [255,222,33,255]

    red_color = [255,0,0,255] 
    green_color = [0,255,0,255]
    uncertain_color = [0,0,255,255]

    grid_red_color = [255, 0, 0, 255]

    orange_color = [255, 140, 0, 255]  # velocity arrow color

    # Set up line series styling for RG beliefs (static, applies to all timelines)
    rr.log("rg_beliefs/red", rr.SeriesLines(colors=[255, 0, 0, 255], names="Red", widths=3), static=True)
    rr.log("rg_beliefs/green", rr.SeriesLines(colors=[0, 255, 0, 255], names="Green", widths=3), static=True)
    rr.log("rg_beliefs/uncertain", rr.SeriesLines(colors=[0, 0, 255, 255], names="Uncertain", widths=3), static=True)

    for timestep in range(len(rgb_video)):
        rr.set_time(timeline = "frame", sequence = timestep)
        if rgb_video_highres is not None:
            rr.log("stimulus/", rr.Image(rgb_video_highres[timestep]))
        else:
            rr.log("stimulus/", rr.Image(rgb_video[timestep]))

        rr.log("jtap/", rr.Image(rgb_video[timestep]))

        rr.log("jtap/tracking_positions", 
            rr.Points2D(
                positions = tracking_uv[timestep],
                colors = black_color,
                radii = tracking_dot_radii[timestep],
                draw_order=50
            )
        )

        rr.log("jtap/prediction_lines", 
            rr.LineStrips2D(
                strips = prediction_uv[timestep],
                colors = yellow_color,
                radii = prediction_line_radii[timestep],
                draw_order=10
            )
        )

        # --- Plot velocity as arrows if enabled ---
        if show_velocity:
            # For each particle, plot velocity vector as an arrow
            # tracking_uv[timestep] shape: (N_PARTICLES, 2)
            # tracking_direction[timestep], tracking_speed[timestep]: each (N_PARTICLES,)
            # Need to compute dx, dy for each using direction (radians) and speed (pixels/scene-units)
            # -- convert speed to displacement in UV space --
            # However, speed is in scene units. Need to project it into (delta u, delta v) per particle
            angles = tracking_direction[timestep]
            speeds = tracking_speed[timestep]  # speeds in scene units per step

            # The vector in scene space is dx = cos(angle) * speed, dy = sin(angle) * speed
            dx_scene = np.cos(angles) * speeds
            dy_scene = np.sin(angles) * speeds

            # Transform the (dx, dy) vector from scene to UV (pixel) units:
            dx_pixels = dx_scene * pixel_density  # The pixel_density is scene-to-pixel scale
            dy_pixels = -dy_scene * pixel_density  # minus because v goes downward in image space

            # (vx, vy) in rerun is (dx_pixel, -dy_pixel)
            vectors = np.stack([dx_pixels, dy_pixels], axis=1)  # shape: (N_PARTICLES, 2)
            origins = tracking_uv[timestep]  # (N_PARTICLES, 2)
            radii = velocity_arrow_radii[timestep]  # (N_PARTICLES,)

            rr.log("jtap/velocity_arrows", rr.Arrows2D(
                origins=origins,
                vectors=vectors,
                colors=orange_color,
                radii=radii,
                draw_order=30
            ))

        # ---- Render grid over scene if enabled ----
        if render_grid and grid_positions_per_frame is not None:
            if has_valid_grid_over_time[timestep]:
                grid_positions = grid_positions_per_frame[timestep]
                if grid_positions.shape[0] > 0:
                    rr.log(
                        "jtap/grid_points",
                        rr.Points2D(
                            positions=grid_positions,
                            colors=grid_red_color,
                            radii=grid_dot_radius,
                            draw_order=100
                        )
                    )
            else:
                rr.log("jtap/grid_points", rr.Clear(recursive=True))

        rr.log("inferred_speed/", 
            rr.BarChart(
                values = speed_bin_counts[timestep],
                # labels = speed_bin_centers
            )
        )
        
        rr.log("inferred_direction/", 
            rr.BarChart(
                values = direction_bin_counts[timestep],
                # labels = direction_bin_centers
            )
        )

        # Log RG beliefs as line series over time
        rr.log("rg_beliefs/red", rr.Scalars(rg_raw_beliefs[timestep, 1]))  # Red is index 1
        rr.log("rg_beliefs/green", rr.Scalars(rg_raw_beliefs[timestep, 0]))  # Green is index 0
        rr.log("rg_beliefs/uncertain", rr.Scalars(rg_raw_beliefs[timestep, 2]))  # Uncertain is index 2
        
        # Format ESS information
        ess_text = f"__{ESS_over_time[timestep]:.2f}__" if ESS_over_time[timestep] <= ESS_threshold else f"{ESS_over_time[timestep]:.2f}"
        resampled_text = f" **RESAMPLED** (Threshold: {ESS_threshold})" if resampled_over_time[timestep] else ""
        
        # Format weight component correlations
        corr_prev = weight_component_correlations['prev'][timestep]
        corr_incr = weight_component_correlations['incr'][timestep]
        corr_prop = weight_component_correlations['prop'][timestep]
        corr_grid = weight_component_correlations['grid'][timestep]
        
        # Find the highest correlation for this timestep
        correlations = {'prev': corr_prev, 'incr': corr_incr, 'prop': corr_prop, 'grid': corr_grid}
        max_corr_key = max(correlations, key=correlations.get)
        
        # Format correlations with bold for highest
        corr_texts = []
        for key, value in correlations.items():
            if key == max_corr_key:
                corr_texts.append(f"**{key}: {value:.3f}**")
            else:
                corr_texts.append(f"{key}: {value:.3f}")
        
        corr_text = " | ".join(corr_texts)
        
        info_text = f"> ESS: {ess_text} / {num_particles}{resampled_text}\n\n> Weight Correlations: {corr_text}"
        rr.log("info/ESS", rr.TextDocument(info_text, media_type=rr.MediaType.MARKDOWN))

def red_green_viz_notebook(JTAP_data, viz_key = jax.random.PRNGKey(0), prediction_t_offset = None, video_offset = (0,0),
    fps = 10, skip_t = 1, show_latents = True, min_dot_alpha = 0.2, min_line_alpha = 0.04, show_resampled_text = True, num_t_steps = None, diameter = 1.0):
    ###
    """
    ASSUMPTIONS: 
    1. SCENE IS STATIC (no changing barriers & occluders)
    2. View is overlayed on observations
    """
    ###

    assert isinstance(JTAP_data, JTAPData), "JTAP_data must be of type JTAPData"
    obs_arrays = JTAP_data.stimulus.discrete_obs
    inference_input = JTAP_data.params.inference_input

    def generate_samples(key, n_samples, support, logprobs):
        keys = jax.random.split(key, n_samples)
        sampled_idxs = jax.vmap(jax.random.categorical, in_axes = (0, None))(keys, logprobs)
        return support[sampled_idxs]

    generate_samples_vmap = jax.vmap(generate_samples, in_axes = (0,None,0,0))

    if prediction_t_offset is None:
        prediction_t_offset = JTAP_data.params.max_prediction_steps

    max_line_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_line_alpha = min_line_alpha

    max_dot_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_dot_alpha = min_dot_alpha

    if num_t_steps is None:
        num_inference_steps = JTAP_data.inference.weight_data.final_weights.shape[0]
    else:
        num_inference_steps = num_t_steps
    num_prediction_steps = JTAP_data.params.max_prediction_steps
    max_inference_T = num_inference_steps - 1
    maxt = obs_arrays.shape[0]
    n_particles = JTAP_data.params.num_particles
    normalized_weights_ = jnp.exp(JTAP_data.inference.weight_data.final_weights - logsumexp(JTAP_data.inference.weight_data.final_weights, axis = 1, keepdims = True)) # T by N_PARTICLES

    equal_prob_value = 1/n_particles

    particle_collapsed = jnp.any(jnp.isnan(normalized_weights_), axis = 1)

    normalized_weights = jnp.where(jnp.isnan(normalized_weights_), jnp.full_like(normalized_weights_, equal_prob_value), normalized_weights_)

    if max_inference_T > maxt:
        print("Too many timesteps in particle data")
        return

    if num_inference_steps - video_offset[0] - video_offset[1] <= 0:
        print(f"Video limits are too restrictive. Plotting from T = {video_offset[0]}"+\
              f" to T = {max_inference_T - video_offset[1]} " +\
              "is not possible")
        return

    if num_prediction_steps < prediction_t_offset:
        print(f"Prediction offset is too high. Prediction offset is {prediction_t_offset}"+\
              f" but max prediction steps is {num_prediction_steps}")
        return
    
    max_inference_T_for_video = max_inference_T - video_offset[1]


    inf_x_points = JTAP_data.inference.tracking.x # T by N_PARTICLES
    inf_y_points = JTAP_data.inference.tracking.y # T by N_PARTICLES
    pred_x_lines = jnp.concatenate([JTAP_data.inference.tracking.x[:,None,:], JTAP_data.inference.prediction.x], axis = 1) # T by T_pred+1 by N_PARTICLES
    pred_y_lines = jnp.concatenate([JTAP_data.inference.tracking.y[:,None,:], JTAP_data.inference.prediction.y], axis = 1) # T by T_pred+1 by N_PARTICLES
    inf_dots_alpha_over_time = min_dot_alpha + normalized_weights * (max_dot_alpha - min_dot_alpha)
    pred_alphas_over_time = min_line_alpha + normalized_weights * (max_line_alpha - min_line_alpha)
    sizes_over_time = jnp.full((num_inference_steps, n_particles), diameter) # T by N_PARTICLES

    color_mapping = {
        0: (255, 255, 255),  # white
        1: (128, 128, 128),  # grey
        2: (0, 0, 255),      # blue
        3: (0, 0, 0),        # black
        4: (255, 0, 0),      # red
        5: (0, 255, 0)       # green
    }
    # Create a list of the RGB colors in the order of their keys
    color_list = [color_mapping[i] for i in range(6)]
    color_array = np.array(color_list, dtype=np.uint8)
    frames = []
    for arr in obs_arrays:
        rgb_array = color_array[np.array(arr, dtype = np.uint8)]
        image = Image.fromarray(rgb_array)
        frames.append(np.array(image))
        
    # Create a figure
    if show_latents:
        fig = plt.figure(figsize=(10,8))
        gs = GridSpec(12, 12, figure=fig)
        ax1 = fig.add_subplot(gs[:7, :7])
        ax2 = fig.add_subplot(gs[1:6, 8:]) 
        ax3 = fig.add_subplot(gs[8:, :7], projection = 'polar')
        ax4 = fig.add_subplot(gs[8:, 8:])  
    else:
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1, 12, figure=fig)
        ax1 = fig.add_subplot(gs[0, :6])
        ax2 = fig.add_subplot(gs[0, 7:]) 

    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_yticks([])
    ax1.set_title('Position (' + r'$S_x, S_y$' +')')

    # AX1

    # WORKS WELL NOW
     
    scale_multiplier = 1 / JTAP_data.params.inference_input.image_discretization
    image_height = obs_arrays.shape[1] # 2nd dimension is height
    x_to_pix_x = jax.jit(lambda x, s : (x + 0.5*s) * scale_multiplier)
    y_to_pix_y = jax.jit(lambda y, s : image_height - 1 - (y + 0.5*s) * scale_multiplier)

    x_to_pix_x_vmap = jax.vmap(x_to_pix_x)
    y_to_pix_y_vmap = jax.vmap(y_to_pix_y)

    filtering_posterior_dot_probs = ax1.scatter(x_to_pix_x_vmap(inf_x_points[0], sizes_over_time[0]), 
        y_to_pix_y_vmap(inf_y_points[0], sizes_over_time[0]),
        s = 20,
        c = 'k', linewidths = 0,
        zorder=5,
        alpha = inf_dots_alpha_over_time[0])
            
    p_lines = []
    for n in range(n_particles):
        p_lines.append(Line2D(x_to_pix_x(pred_x_lines[0,:,n], sizes_over_time[0,n]), 
            y_to_pix_y(pred_y_lines[0,:,n], sizes_over_time[0,n]), color='orange', 
            alpha=round(float(pred_alphas_over_time[0,n]),2), zorder=4, linestyle="-"))
        ax1.add_line(p_lines[n])
        
    im = ax1.imshow(frames[0])

    timer_text = fig.text(0.1, 0.95, "Timestep: " + r'$\bf{0}$', ha='left', color="k", fontsize=15)
    if show_resampled_text:
        resampled_text = fig.text(0.4, 0.95, f"Resampled: {JTAP_data.inference.resampled[0]}", ha='left', color="b", fontsize=17)

    particle_collapsed_text = fig.text(0.7, 0.95, f"Particle Collapsed: {particle_collapsed[0]}", ha='left', color="r", fontsize=15)

    # AX2
    rg_data = get_rg_raw_beliefs(JTAP_data, prediction_t_offset)
    bars = ax2.bar(range(3), rg_data[0], color = ['green', 'red', 'blue'])

    # Set up the axis limits
    # Set title and x-ticks
    ax2.set_title('Red or Green: Which will it hit next?', fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Green', 'Red', 'Uncertain'])
    ax2.set_ylim(0, 1)

    if show_latents:
        # AX3
        num_bins = 90
        n_samples = 10000
        max_count = 10
        viz_key, dir_key = jax.random.split(viz_key, 2)
        dir_keys = jax.random.split(dir_key, num_inference_steps)
        sampled_dir = generate_samples_vmap(dir_keys, 10000, JTAP_data.inference.tracking.direction[:num_inference_steps], JTAP_data.inference.weight_data.final_weights[:num_inference_steps])
        all_counts_dir = []
        # NOTE: IN THIS VIZ, STEP DIR IS TAKEN  a step before current step
        for i in range(max_inference_T_for_video + 1):
            counts_dir, bin_edges_dir = np.histogram(sampled_dir[i], bins=num_bins, range=(-np.pi, np.pi), density=True)
            # rescale to sqrt of counts for it to be compatible with volume of pie histogram
            counts_dir = np.sqrt(counts_dir)
            all_counts_dir.append(counts_dir * (max_count / max(counts_dir)))

        bin_centers = (bin_edges_dir[:-1] + bin_edges_dir[1:]) / 2
        
        # Plot the histogram on the polar plot
        dir_bars = ax3.bar(bin_centers, all_counts_dir[0], width=bin_edges_dir[1] - bin_edges_dir[0], bottom=0, color='brown', edgecolor=None, alpha=0.6)
        ax3.grid(False)
        ax3.set_title('Direction (' + r'$\phi$' +')')
        ax3.set_ylim(0, max_count)
        ax3.set_yticklabels([])
        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)
        ax3.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])

       # AX4
        num_bins = int(inference_input.max_speed/ 0.1) + 1
        n_samples = 10000
        max_count = 10
        viz_key, speed_key = jax.random.split(viz_key, 2)
        speed_keys = jax.random.split(speed_key, num_inference_steps)
        sampled_speed = generate_samples_vmap(speed_keys, 10000, JTAP_data.inference.tracking.speed[:num_inference_steps], JTAP_data.inference.weight_data.final_weights[:num_inference_steps])
        all_counts_speed = []
        # NOTE: IN THIS VIZ, STEP SPEED IS TAKEN  a step before current step
        for i in range(max_inference_T_for_video + 1):
            counts_speed, bin_edges_speed = np.histogram(sampled_speed[i], bins=num_bins, range=(0, inference_input.max_speed), density=True)
            all_counts_speed.append(counts_speed * (max_count / sum(counts_speed)))

        bin_centers = (bin_edges_speed[:-1] + bin_edges_speed[1:]) / 2
        speed_bars = ax4.bar(bin_centers, all_counts_speed[0], width=bin_edges_speed[1] - bin_edges_speed[0], bottom=0, color='orange', edgecolor=None, alpha=0.6)
        # ax4.grid(False)
        ax4.set_title('Speed (' + r'$\nu$' +')')
        ax4.set_ylim(0, max_count)
        ax4.set_yticks([])    
        ax4.set_yticklabels([])

    def init_func():
        pass

    def update(frame_idx):
        # if idx <= video_offset[0]:
        #     return
        frame_idx += skip_t*(video_offset[0])
        im.set_array(frames[frame_idx])
        if frame_idx % skip_t == 0:
            idx = int(frame_idx/skip_t)
            timer_text.set_text(f"Timestep: " + r'$\bf{' + str(idx) + r'}$')
            particle_collapsed_text.set_text(f"Particle Collapsed: {particle_collapsed[idx]}")
            if show_resampled_text:
                if particle_collapsed[idx]:
                    resampled_text.set_color("r")
                    resampled_text.set_text("Resampled: Disabled")
                else:
                    if JTAP_data.inference.resampled[idx]:
                        resampled_text.set_color("g")
                    else:
                        resampled_text.set_color("b")
                    resampled_text.set_text(f"Resampled: {JTAP_data.inference.resampled[idx]}")
            print(f"rendering inference step {idx}")
            filtering_posterior_dot_probs.set_offsets(np.c_[
                x_to_pix_x_vmap(inf_x_points[idx], 
                    sizes_over_time[idx]),
                y_to_pix_y_vmap(inf_y_points[idx], 
                    sizes_over_time[idx])
            ])
            filtering_posterior_dot_probs.set_alpha(inf_dots_alpha_over_time[idx])
            for n in range(n_particles):

                p_lines[n].set_data(x_to_pix_x(pred_x_lines[idx,:,n], sizes_over_time[idx,n]), 
                    y_to_pix_y(pred_y_lines[idx,:,n], sizes_over_time[idx,n]))
                p_lines[n].set_alpha(round(float(pred_alphas_over_time[idx,n]),2))

            for bar, height in zip(bars, rg_data[idx]):
                bar.set_height(height)
            if show_latents:
                for bar, height in zip(dir_bars, all_counts_dir[idx]):
                    bar.set_height(height)
                for bar, height in zip(speed_bars, all_counts_speed[idx]):
                    bar.set_height(height)

    n_frames = max_inference_T_for_video + 1 - video_offset[0]
    animation = FuncAnimation(fig, update, frames=skip_t*n_frames, init_func=init_func, interval=1000//fps)
    plt.close()
    return HTML_Display(animation.to_html5_video())