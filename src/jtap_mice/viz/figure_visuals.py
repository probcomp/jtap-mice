import jax
import jax.numpy as jnp
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import seaborn as sns
from jax.scipy.special import logsumexp
from jtap_mice.evaluation import get_rg_raw_beliefs, JTAP_Metrics, CombinedJTAP_Metrics, DecisionMetrics, CombinedDecisionMetrics
from jtap_mice.inference import JTAPMiceData
from jtap_mice.utils import discrete_obs_to_rgb, d2r, r2d

def interpretable_belief_viz(JTAP_data, high_res_video, prediction_t_offset=5, video_offset=(0, 0), viz_key=jax.random.PRNGKey(0), 
                  min_dot_alpha=0.2, min_line_alpha=0.04,
                  ax_num=None, timeframe=None, high_res_multiplier=5, diameter=1.0):

    assert isinstance(JTAP_data, JTAPMiceData), "JTAP_data must be of type JTAPMiceData"

    inference_input = JTAP_data.params.inference_input

    def generate_samples(key, n_samples, support, logprobs):
        keys = jax.random.split(key, n_samples)
        sampled_idxs = jax.vmap(jax.random.categorical, in_axes = (0, None))(keys, logprobs)
        return support[sampled_idxs]

    generate_samples_vmap = jax.vmap(generate_samples, in_axes = (0, None, None, None))

    max_line_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_line_alpha = min_line_alpha

    max_dot_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_dot_alpha = min_dot_alpha

    num_inference_steps = JTAP_data.inference.weight_data.final_weights.shape[0]
    num_prediction_steps = JTAP_data.params.max_prediction_steps
    max_inference_T = num_inference_steps - 1
    maxt = high_res_video.shape[0]
    n_particles = JTAP_data.params.num_particles
    normalized_weights = jnp.exp(JTAP_data.inference.weight_data.final_weights - logsumexp(JTAP_data.inference.weight_data.final_weights, axis = 1, keepdims = True)) # T by N_PARTICLES

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
    frames = high_res_video
        
    # Create a figure

    fig = plt.figure(figsize=(10,8), dpi=300)
    gs = GridSpec(12, 12, figure=fig)
    ax1 = fig.add_subplot(gs[:7, :7])
    ax2 = fig.add_subplot(gs[1:6, 8:]) 
    ax3 = fig.add_subplot(gs[8:, :7], projection = 'polar')
    ax4 = fig.add_subplot(gs[8:, 8:])  


    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_yticks([])
    # ax1.set_title('Position (' + r'$S_x, S_y$' +')')

    # AX1

    # WORKS WELL NOW
    im_height = frames[0].shape[0]
    scale_multiplier = 1 / (JTAP_data.params.image_discretization)*high_res_multiplier
    x_to_pix_x = jax.jit(lambda x, s : (x + 0.5*s) * scale_multiplier)
    y_to_pix_y = jax.jit(lambda y, s : im_height - 1 - (y + 0.5*s) * scale_multiplier)

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

    # AX2
    rg_data = get_rg_raw_beliefs(JTAP_data, prediction_t_offset)
    bars = ax2.bar(range(3), rg_data[0], color = ['green', 'red', 'blue'])

    # Set up the axis limits
    # Set title and x-ticks
    ax2.set_title('Red or Green: Which will it hit next?', fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Green', 'Red', 'Uncertain'])
    ax2.set_ylim(0, 1)

    # AX3
    num_bins = 90
    n_samples = 10000
    max_count = 10
    viz_key, dir_key = jax.random.split(viz_key, 2)
    dir_keys = jax.random.split(dir_key, num_inference_steps)
    sampled_dir = generate_samples_vmap(dir_keys, n_samples, JTAP_data.inference.tracking.direction, JTAP_data.inference.weight_data.final_weights)
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
    # ax3.set_title('Direction (' + r'$\phi$' +')')
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
    sampled_speed = generate_samples_vmap(speed_keys, n_samples, JTAP_data.inference.tracking.speed, JTAP_data.inference.weight_data.final_weights)
    all_counts_speed = []
    # NOTE: IN THIS VIZ, STEP SPEED IS TAKEN  a step before current step
    for i in range(max_inference_T_for_video + 1):
        counts_speed, bin_edges_speed = np.histogram(sampled_speed[i], bins=num_bins, range=(0, inference_input.max_speed), density=True)
        all_counts_speed.append(counts_speed * (max_count / sum(counts_speed)))

    bin_centers = (bin_edges_speed[:-1] + bin_edges_speed[1:]) / 2
    speed_bars = ax4.bar(bin_centers, all_counts_speed[0], width=bin_edges_speed[1] - bin_edges_speed[0], bottom=0, color='orange', edgecolor=None, alpha=0.6)
    # ax4.grid(False)
    # ax4.set_title('Speed (' + r'$\nu$' +')')
    ax4.set_ylim(0, max_count)
    ax4.set_yticks([])    
    ax4.set_yticklabels([])

    # Function to update all axes for a given timeframe
    def update_frame(idx):
        im.set_array(frames[idx])
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
        for bar, height in zip(dir_bars, all_counts_dir[idx]):
            bar.set_height(height)
        for bar, height in zip(speed_bars, all_counts_speed[idx]):
            bar.set_height(height)

        return ax1, ax2, ax3, ax4

    # If a specific timeframe and ax are requested
    if timeframe is not None and ax_num is not None:
        if timeframe < 0 or timeframe >= maxt:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be in range [0, {maxt - 1}]")

        all_axes = [ax1, ax2, ax3, ax4]
        update_frame(timeframe)
        # Hide all other axes except the requested one
        for i, ax in enumerate(all_axes):
            if i != ax_num:
                ax.set_visible(False)

        # Return the requested axis
        return fig, all_axes[ax_num]
    # If no specific ax/timeframe is requested, raise an error
    raise ValueError("You must specify both ax_num and timeframe to return a specific ax.")


def create_log_frequency_heatmaps(jtap_metrics, targeted_analysis=False, bins=25, 
                                  weighted=False, cmap='Oranges', cmap_reverse=False, 
                                  model_name='JTAP'):
    """
    Create log-frequency heatmaps comparing human and model decision patterns.
    
    Args:
        jtap_metrics: JTAP_Metrics or CombinedJTAP_Metrics object containing decision metrics
        targeted_analysis: If True, plot all three models in 3x2 grid; if False, plot only 'model' in 1x2 grid
        bins: Number of bins for the heatmap (default: 20)
        weighted: Whether to use correlation weights (default: False)
        cmap: Matplotlib colormap name (default: 'viridis')
        cmap_reverse: Whether to reverse the colormap (default: False)
        model_name: Name to display for the main model (default: 'JTAP')
    
    Returns:
        matplotlib.figure.Figure: The complete figure with all heatmaps
    """
    
    # Check if it's either JTAP_Metrics or CombinedJTAP_Metrics
    if isinstance(jtap_metrics, JTAP_Metrics):
        assert isinstance(jtap_metrics.model_metrics, DecisionMetrics), "jtap_metrics must contain a DecisionMetrics object for the model"
        assert isinstance(jtap_metrics.frozen_metrics, DecisionMetrics), "jtap_metrics must contain a DecisionMetrics object for the frozen model"
        assert isinstance(jtap_metrics.decaying_metrics, DecisionMetrics), "jtap_metrics must contain a DecisionMetrics object for the decaying model"
    elif isinstance(jtap_metrics, CombinedJTAP_Metrics):
        assert isinstance(jtap_metrics.model_metrics, CombinedDecisionMetrics), "jtap_metrics must contain a CombinedDecisionMetrics object for the model"
        assert isinstance(jtap_metrics.frozen_metrics, CombinedDecisionMetrics), "jtap_metrics must contain a CombinedDecisionMetrics object for the frozen model"
        assert isinstance(jtap_metrics.decaying_metrics, CombinedDecisionMetrics), "jtap_metrics must contain a CombinedDecisionMetrics object for the decaying model"
    else:
        raise TypeError("jtap_metrics must be either a JTAP_Metrics or CombinedJTAP_Metrics object")
    
    # Select the appropriate analysis data based on targeted_analysis flag
    def get_analysis_data(metrics):
        """Get the appropriate analysis data based on targeted_analysis flag."""
        if targeted_analysis:
            return metrics.targeted
        else:
            return metrics.non_targeted

    # Set style
    plt.style.use('default')
    sns.set_style("white")
    
    # Determine layout based on targeted_analysis
    if targeted_analysis:
        # Full 3x2 grid for all three models
        fig = plt.figure(figsize=(14, 22))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.15)
        model_types = ["model", "frozen", "decaying"]
        titles = [model_name, 'Frozen', 'Decaying']
    else:
        # Single 1x2 grid for model only
        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(1, 2, figure=fig, hspace=0.7, wspace=0.15)
        model_types = ["model"]
        titles = [model_name]
    
    # Define bin edges
    bin_edges = np.linspace(0, 1, bins + 1)
    
    def calculate_heatmap(human_data, model_data, correl_weights):
        """Calculate log-frequency heatmap for given data."""
        # Digitize both arrays into bins
        human_bin_idx = np.digitize(human_data, bin_edges) - 1
        human_bin_idx[human_bin_idx == bins] = bins - 1
        model_bin_idx = np.digitize(model_data, bin_edges) - 1
        model_bin_idx[model_bin_idx == bins] = bins - 1
        
        # Initialize the 2D histogram
        heatmap = np.zeros((bins, bins))
        heatmap_weight = np.zeros((bins, bins))
        
        # Accumulate counts
        for h, m, weight in zip(human_bin_idx, model_bin_idx, correl_weights):
            heatmap[h, m] += 1
            if targeted_analysis:
                heatmap_weight[h, m] += weight / 59  # Normalize as in original
            else:
                heatmap_weight[h, m] += weight
        
        # Compute log frequencies
        if targeted_analysis:
            log_heatmap = np.log1p(heatmap)
        else:
            # Normalize weights for model-only version
            if np.sum(heatmap_weight) > 0:
                heatmap_weight /= np.sum(heatmap_weight)
            log_heatmap = np.log1p(heatmap) + np.log1p(heatmap_weight)
        
        return log_heatmap
    
    # Extract data for all required models
    all_heatmaps_pressed = []
    all_heatmaps_green = []
    all_pressed_correl = []
    all_green_correl = []
    
    for i, model_type in enumerate(model_types):
        metrics = jtap_metrics.model_metrics if model_type == "model" else jtap_metrics.frozen_metrics if model_type == "frozen" else jtap_metrics.decaying_metrics
        
        # Get the appropriate analysis data based on targeted_analysis flag
        analysis_data = get_analysis_data(metrics)
        
        # Extract decision probabilities (P(Red or Green))
        human_decision_probs = analysis_data.human_decision_probs[analysis_data.valid_decision_mask]
        model_decision_probs = analysis_data.model_decision_probs[analysis_data.valid_decision_mask]
        
        # Extract conditional green probabilities (P(Green | Red or Green))
        human_green_given_decision = analysis_data.human_green_given_decision[analysis_data.valid_conditional_mask]
        model_green_given_decision = analysis_data.model_green_given_decision[analysis_data.valid_conditional_mask]
        decision_weights = analysis_data.decision_weights[analysis_data.valid_conditional_mask]
        
        # Set up correlation weights
        if weighted:
            correl_weights_green = decision_weights
        else:
            correl_weights_green = np.zeros_like(human_green_given_decision)
        
        # Always use uniform weights for decision probabilities
        correl_weights_pressed = np.zeros_like(human_decision_probs)
        
        # Calculate heatmaps
        heatmap_pressed = calculate_heatmap(human_decision_probs, model_decision_probs, correl_weights_pressed)
        heatmap_green = calculate_heatmap(human_green_given_decision, model_green_given_decision, correl_weights_green)
        
        all_heatmaps_pressed.append(heatmap_pressed)
        all_heatmaps_green.append(heatmap_green)
        
        # Calculate correlations
        pressed_corr = pearsonr(human_decision_probs, model_decision_probs)[0]
        all_pressed_correl.append(pressed_corr)
        
        # Use weighted correlation from metrics
        green_corr = analysis_data.weighted_conditional_green_corr if weighted else analysis_data.conditional_green_corr
        all_green_correl.append(green_corr)
    
    # Set up colormaps
    if cmap_reverse:
        cmap = plt.get_cmap(cmap).reversed()
    else:
        cmap = plt.get_cmap(cmap)
    
    # Determine global color scale
    vmin = min(np.min(hp) for hp in all_heatmaps_pressed + all_heatmaps_green)
    vmax = max(np.max(hp) for hp in all_heatmaps_pressed + all_heatmaps_green)
    
    # Column headers
    column_headers = ['$\\mathbf{P(Decision)}$', '$\\mathbf{P(Green \\; | \\; Decision)}$']
    
    # Plot heatmaps
    for row, (title, hp, hg, pressed_corr, green_corr) in enumerate(
        zip(titles, all_heatmaps_pressed, all_heatmaps_green, all_pressed_correl, all_green_correl)):
        
        # Left heatmap (P(Red or Green))
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(hp, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        
        # Right heatmap (P(Green | Red or Green))
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(hg, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        
        # Styling based on layout
        if targeted_analysis:
            # Labels for 3x2 layout
            if row == 0:
                fig.text(0.5, 0.64, title, ha='center', va='center', 
                         fontsize=26, fontweight='bold')
            elif row == 1:
                fig.text(0.5, 0.355, title, ha='center', va='center', 
                         fontsize=26, fontweight='bold')
            elif row == 2:
                fig.text(0.5, 0.085, title, ha='center', va='center', 
                         fontsize=26, fontweight='bold')
            
            ax1.set_ylabel('Participants', fontsize=22, fontweight='bold')
            ax2.set_ylabel('Participants', fontsize=22, fontweight='bold')
            ax1.set_title(f"$r = {pressed_corr:.2f}$", fontsize=24, fontweight='bold', pad=10)
            green_corr_label = f"$r_{{\\text{{wtd}}}} = {green_corr:.2f}$" if weighted else f"$r = {green_corr:.2f}$"
            ax2.set_title(green_corr_label, fontsize=24, fontweight='bold', pad=10)
        else:
            # Labels for 1x2 layout
            ax1.set_xlabel(title, fontsize=24, fontweight='bold', labelpad=15)
            ax2.set_xlabel(title, fontsize=24, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Participants', fontsize=24, fontweight='bold', labelpad=15)
            ax1.text(0.01, 0.925, f"$r = {pressed_corr:.2f}$", fontsize=24, fontweight='bold', transform=ax1.transAxes)
            green_corr_label = f"$r_{{\\text{{wtd}}}} = {green_corr:.2f}$" if weighted else f"$r = {green_corr:.2f}$"
            ax2.text(0.01, 0.925, green_corr_label, fontsize=24, fontweight='bold', transform=ax2.transAxes)
            ax1.set_title(column_headers[0], fontsize=24, fontweight='bold', pad=20)
            ax2.set_title(column_headers[1], fontsize=24, fontweight='bold', pad=20)
        
        # Grid settings
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            # Add spines styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
            # Improve tick appearance
            ax.tick_params(width=1.5, length=6, color='#333333', labelsize=14)
    
    # Add colorbar
    if targeted_analysis:
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Log Frequency', shrink=0.5)
        cbar.set_label('Log Frequency', fontsize=20)
        cbar.ax.tick_params(labelsize=14)
        
        # Add column headers
        fig.text(0.3, 0.91, column_headers[0], ha='center', va='center', 
                 fontsize=24, fontweight='bold')
        fig.text(0.7, 0.91, column_headers[1], ha='center', va='center', 
                 fontsize=24, fontweight='bold')
        
        # Add horizontal lines
        h_line1 = Line2D([0.1, 0.9], [0.345, 0.345], color='black', linestyle='-', linewidth=1)
        fig.add_artist(h_line1)
        h_line2 = Line2D([0.1, 0.9], [0.625, 0.625], color='black', linestyle='-', linewidth=1)
        fig.add_artist(h_line2)
    else:
        cbar_ax = fig.add_axes([0.35, 0.025, 0.3, 0.04])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Log Frequency')
        cbar.set_label('Log Frequency', fontsize=20)
        cbar.ax.tick_params(labelsize=14)
    
    return fig

def draw_stimulus_image(
    jtap_stimulus,
    frame=0,
    FPS=None,
    line_color=(0, 0, 0),
    line_thickness=0.5,
    text_color=(0, 0, 0),
    font_scale=1.0,
    font_thickness=0.5,
    marker_color=(255, 140, 0),  # more orange
    marker_radius=4,
    marker_border_thickness=1,
    show_blue_dotted_ring=True  # NEW ARGUMENT
):
    # Extract data from jtap_stimulus
    ground_truth_positions = jtap_stimulus.ground_truth_positions
    discrete_obs = jtap_stimulus.discrete_obs
    stimulus_fps = jtap_stimulus.fps
    skip_t = jtap_stimulus.skip_t

    # Use provided FPS or default to stimulus FPS
    if FPS is None:
        FPS = stimulus_fps

    # Calculate effective FPS accounting for skipped frames
    effective_fps = stimulus_fps / skip_t

    # Convert discrete observation to RGB for the specified frame using the existing function
    frame_discrete_obs = discrete_obs[frame:frame + 1]  # Keep batch dimension for function
    rgb_frame = discrete_obs_to_rgb(frame_discrete_obs)[0]  # Get first (and only) frame

    upscale_factor = 4
    original_height, original_width = rgb_frame.shape[:2]
    upscale_height, upscale_width = original_height * upscale_factor, original_width * upscale_factor
    high_res_image = cv2.resize(rgb_frame, (upscale_width, upscale_height), interpolation=cv2.INTER_NEAREST)

    # Get diameter, pixel_density, and image_height for uv conversion
    diameter = getattr(jtap_stimulus, "diameter", 1)
    pixel_density = getattr(jtap_stimulus, "pixel_density", 1)
    image_height = original_height

    # xy_to_uv as in jtap_viz.py (no upscaling!)
    xy_to_uv = lambda x, y: (
        (x + 0.5 * diameter) * pixel_density,
        image_height - ((y + 0.5 * diameter) * pixel_density)
    )

    # Convert ground truth positions to pdata format for compatibility
    pdata = {}
    for i, (x, y) in enumerate(ground_truth_positions):
        pdata[i] = {'x': x, 'y': y}

    sorted_frames = sorted(pdata.keys())

    # Draw trajectory - in original pixel coordinates, then upscale
    for i in range(1, len(sorted_frames)):
        prev_frame = sorted_frames[i - 1]
        current_frame = sorted_frames[i]

        # Convert world coordinates to image coordinates using xy_to_uv
        x1_uv, y1_uv = xy_to_uv(pdata[prev_frame]['x'], pdata[prev_frame]['y'])
        x2_uv, y2_uv = xy_to_uv(pdata[current_frame]['x'], pdata[current_frame]['y'])

        # Upscale for high res image
        x1 = x1_uv * upscale_factor
        y1 = y1_uv * upscale_factor
        x2 = x2_uv * upscale_factor
        y2 = y2_uv * upscale_factor

        # Draw line segment for the trajectory
        cv2.line(
            high_res_image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            line_color,
            int(line_thickness * upscale_factor),  # Adjust thickness for high resolution
            lineType=cv2.LINE_AA
        )

    def find_clear_position(img, x, y, text, search_radius=20):
        """Find a nearby position that avoids dark pixels, considering text size."""
        h, w, _ = img.shape
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, int(font_thickness * upscale_factor))[0]

        for r in range(1, search_radius):
            for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r)]:
                nx, ny = int(x + dx), int(y + dy)
                x_end, y_end = nx + (text_size[0] + 3), ny - (text_size[1] + 3)

                if 0 <= nx < w and 0 <= ny < h and 0 <= x_end < w and 0 <= y_end < h:
                    # Check if the entire text bounding box avoids dark pixels
                    region = img[ny - (text_size[1] + 3):ny, nx:x_end]
                    if not np.any(np.all(region < 27, axis=-1)):
                        return nx, ny
        return x, y  # Default to the original position if no clear spot is found

    # Annotate seconds on the trajectory, accounting for effective FPS
    frames_per_second_annotation = effective_fps

    # Instead of looping over existing frames only, compute exact whole seconds,
    # and interpolate if the annotation frame doesn't land exactly on an existing frame.
    max_time_seconds = (max(sorted_frames)) / effective_fps
    num_whole_seconds = int(np.floor(max_time_seconds)) + 1  # Ensure we include last second if possible

    for s in range(num_whole_seconds):
        # Compute the (possibly non-integer) frame that corresponds to this whole second.
        target_frame = s * effective_fps
        text = str(s)

        # Find nearest frames before and after the target
        frame_before = int(np.floor(target_frame))
        frame_after = int(np.ceil(target_frame))

        if frame_before == frame_after and frame_before in pdata:
            # Exact frame exists
            interp_x = pdata[frame_before]['x']
            interp_y = pdata[frame_before]['y']
            point_frame = frame_before
        elif frame_before in pdata and frame_after in pdata and frame_before != frame_after:
            # Interpolate position
            x0, y0 = pdata[frame_before]['x'], pdata[frame_before]['y']
            x1_, y1_ = pdata[frame_after]['x'], pdata[frame_after]['y']
            t = (target_frame - frame_before) / (frame_after - frame_before)
            interp_x = x0 + t * (x1_ - x0)
            interp_y = y0 + t * (y1_ - y0)
            # For purposes of frame match, choose the nearest int
            point_frame = frame_before if abs(target_frame - frame_before) < abs(target_frame - frame_after) else frame_after
        elif frame_before in pdata:
            interp_x = pdata[frame_before]['x']
            interp_y = pdata[frame_before]['y']
            point_frame = frame_before
        elif frame_after in pdata:
            interp_x = pdata[frame_after]['x']
            interp_y = pdata[frame_after]['y']
            point_frame = frame_after
        else:
            # Can't plot this second, skip
            continue

        # Convert (interp_x, interp_y) to (uv) using xy_to_uv, then upscale
        x_uv, y_uv = xy_to_uv(interp_x, interp_y)
        x = x_uv * upscale_factor
        y = y_uv * upscale_factor

        # Draw a marker (circle) at each second's ball center location
        # First, draw a contrasting border (e.g., black) for visibility
        cv2.circle(
            high_res_image,
            (int(round(x)), int(round(y))),
            int(marker_radius * upscale_factor // 4 + marker_border_thickness),
            (0, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA
        )
        # Inner colored marker
        cv2.circle(
            high_res_image,
            (int(round(x)), int(round(y))),
            int(marker_radius * upscale_factor // 4),
            marker_color,
            thickness=-1,
            lineType=cv2.LINE_AA
        )

        # Draw blue dotted ring if requested, but DO NOT for the rendered frame itself
        if show_blue_dotted_ring and (point_frame != frame):
            # Use jtap_stimulus.diameter (which is the BALL DIAMETER in world units)
            ball_diameter_world = getattr(jtap_stimulus, 'diameter', None)
            if ball_diameter_world is not None:
                # BALL RADIUS in original image pixels (NO upscaling here)
                ball_radius_pixels = (ball_diameter_world / 2.0) * pixel_density
            else:
                # fallback: make the ring equal to marker radius in the original image
                ball_radius_pixels = marker_radius

            ring_radius = int(round(ball_radius_pixels * upscale_factor))
            ring_color = (0, 0, 255)  # OpenCV: BGR (pure blue)

            # Make the ring thicker and more visible for larger ball radii
            num_dots = min(16, int(2 * np.pi * ring_radius / 3))  # Dots every ~3px
            dot_radius = 1

            for i in range(num_dots):
                theta = 2 * np.pi * i / num_dots
                cx = int(round(x + ring_radius * np.cos(theta)))
                cy = int(round(y + ring_radius * np.sin(theta)))
                cv2.circle(
                    high_res_image,
                    (cx, cy),
                    dot_radius,
                    ring_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )

        # Find a clear position to place the text
        x_clear, y_clear = find_clear_position(high_res_image, x, y - 15, text)
        y_adjust = 0

        # Annotate the frame number as seconds
        cv2.putText(
            high_res_image,
            text,  # Time in seconds as text
            (int(x_clear), int(y_clear - y_adjust)),  # Adjusted position
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,  # Larger font scale for high resolution
            text_color,
            thickness=int(font_thickness * upscale_factor),  # Adjust thickness for high resolution
            lineType=cv2.LINE_AA
        )

    # Convert to PIL Image
    pil_image = Image.fromarray(high_res_image)

    return pil_image

def plot_proposal_direction_outlier_pdf(
        tau_deg, alpha, 
        figsize=(8, 4.8), 
        linewidth=2.2, 
        font_scale=1.1,
        tau_color="#D7263D", 
        curve_color="#2768bc",
        show_horizontal_dotted=True,
        horizontal_y=0.5,
        horizontal_color="black",
        horizontal_style=":",
        horizontal_width=1.75,
    ):
    """
    Plots a visually refined sigmoid cutoff for proposal direction outliers.

    Parameters:
        tau_deg (float): Cutoff threshold in degrees.
        alpha (float): Sharpness parameter for the sigmoid.
        figsize (tuple): Size of the figure (width, height).
        linewidth (float): Width of the main plotted line.
        font_scale (float): Multiplier for basic font sizes for prettiness.
        tau_color (str): Color for the tau threshold indicator.
        curve_color (str): Color for the sigmoid curve.
        show_horizontal_dotted (bool): If True, draw the horizontal dotted line.
        horizontal_y (float): Y position for the dotted line.
        horizontal_color (str): Color for the horizontal dotted line.
        horizontal_style (str): Line style for the horizontal dotted line.
        horizontal_width (float): Line width for the horizontal dotted line.
    """
    import matplotlib as mpl

    # Choose a harmonious font/scale
    base_font = 12 * font_scale
    labelsize = base_font
    ticksize = int(base_font * 0.90)
    titlesize = int(base_font * 1.44)
    legendsize = int(base_font * 0.90)
    mpl.rcParams.update({
        "axes.titlesize": titlesize,
        "axes.labelsize": labelsize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
        "axes.edgecolor": "#888",
        "axes.linewidth": 1.4,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "legend.fontsize": legendsize
    })

    fig, ax = plt.subplots(figsize=figsize)
    # Light gray background for elegance
    ax.set_facecolor("#fafbfc")
    fig.patch.set_facecolor("#fafbfc")
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=1, alpha=0.19)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.6, alpha=0.1)
    
    tau_rad = d2r(tau_deg)
    angular_vals = jnp.linspace(-np.pi + np.finfo(float).eps, np.pi - np.finfo(float).eps, 9000)
    probs_ = jax.nn.sigmoid(alpha * (jnp.abs(angular_vals) - tau_rad))

    # Plot the main sigmoid curve, slightly muted (subtle shadow for depth)
    ax.plot(r2d(angular_vals), probs_, color=curve_color, linewidth=linewidth, label=None, zorder=10)
    ax.plot(r2d(angular_vals), probs_, color="#23486d", linewidth=linewidth/6, alpha=0.13, zorder=6)

    # Plot vertical tau thresholds with nice styling
    ax.axvline(x=tau_deg, color=tau_color, linestyle="--", linewidth=1.75, alpha=0.95, zorder=12)
    ax.axvline(x=-tau_deg, color=tau_color, linestyle="--", linewidth=1.75, alpha=0.95, zorder=12)

    # Plot horizontal black dotted line, if requested
    if show_horizontal_dotted:
        # Make sure it's visible and on top
        ax.axhline(
            y=horizontal_y, color=horizontal_color,
            linestyle=horizontal_style, linewidth=horizontal_width, alpha=0.98, zorder=21
        )

    # Plot thin dotted lines from tau intersection to y axis (both tau_deg and -tau_deg)
    # tau_deg intersection
    y_tau = float(jax.nn.sigmoid(alpha * (tau_rad - tau_rad)))
    ax.plot([tau_deg, tau_deg], [0, y_tau], color=tau_color, linestyle="-", linewidth=1.0, alpha=0.7, zorder=15)
    # -tau_deg intersection
    ax.plot([-tau_deg, -tau_deg], [0, y_tau], color=tau_color, linestyle=":", linewidth=1.0, alpha=0.7, zorder=15)
    # Optionally, mark the intersection point with a subtle marker:
    ax.scatter([tau_deg, -tau_deg], [y_tau, y_tau], s=44, color=tau_color, alpha=0.55, edgecolors='none', zorder=14)

    # Fill under the curve for visual balance
    ax.fill_between(r2d(angular_vals), 0, probs_, color=curve_color, alpha=0.11, zorder=2)

    # Axis labeling and limits with sanity margins
    ax.set_xlabel("Angular Difference (Predicted vs. Proposed) [degrees]", fontsize=labelsize)
    ax.set_ylabel("Probability", fontsize=labelsize)

    # Tau, alpha in the title (as requested). \\, degree symbol.
    ax.set_title(
        f"Proposal Outlier Sigmoid:  $\\tau$ = {tau_deg:.0f}°, $\\alpha$ = {alpha:g}", 
        fontsize=titlesize, weight="medium", pad=20
    )

    ax.set_ylim(-0.045, 1.11)
    ax.set_xlim(-180, 180)
    
    # Custom ticks for professional style
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(45))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
    ax.tick_params(axis='x', which='both', length=5, width=1)
    ax.tick_params(axis='y', which='both', length=4, width=1)
    
    # Remove frame top/right for aesthetic
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_linewidth(1.4)
    ax.spines['bottom'].set_linewidth(1.4)

    # Annotate tau lines
    ax.text(
        tau_deg + 2, 0.96, r"$+\tau$", color=tau_color, fontsize=labelsize, va='top', ha='left', fontweight='semibold'
    )
    ax.text(
        -tau_deg - 2, 0.96, r"$-\tau$", color=tau_color, fontsize=labelsize, va='top', ha='right', fontweight='semibold'
    )

    # Remove clutter: no legend, prettify
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.88)
    plt.show()