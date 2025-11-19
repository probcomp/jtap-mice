import copy
import cv2
import os
import pickle
import warnings
import pandas as pd

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import seaborn as sns
from jax.scipy.special import logsumexp
from matplotlib import rcParams, font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import (
    t, ttest_ind, ttest_1samp, pearsonr, truncnorm,
)
from tqdm import tqdm


# to keep observations a fixed size for the sake of JITTed inference
def pad_obs_with_last_frame(observations, target_num_frames):
    """
    Pad an observation sequence to a target length by repeating the last frame.
    
    This function extends a sequence of observations to a fixed length by duplicating
    the final frame. This is useful for creating fixed-size arrays for JIT compilation
    while preserving the temporal structure of the original sequence.
    
    Args:
        observations: Array of shape (num_frames, ...) containing the observation sequence
        target_num_frames: Target length for the padded sequence
        
    Returns:
        Array of shape (target_num_frames, ...) with the original observations followed
        by repeated copies of the last frame
        
    Raises:
        AssertionError: If target_num_frames is not greater than the current sequence length
    """
    assert target_num_frames > observations.shape[0], "target_num_frames must be greater than the number of frames in the array"
    
    num_padding_frames = target_num_frames - observations.shape[0]
    last_frame = observations[-1]
    padding_frames = jnp.repeat(last_frame[jnp.newaxis, ...], num_padding_frames, axis=0)
    padded_observations = jnp.concatenate([observations, padding_frames], axis=0)
    
    return padded_observations



@jax.jit
def get_model_score(keypress_data, rg_outcome):
    num_frames = keypress_data.shape[0]
    penalty_outcome = jnp.where(rg_outcome == 1, 0,1)
    score = 20 + 100*((jnp.sum(keypress_data == rg_outcome)/num_frames) - (jnp.sum(keypress_data == penalty_outcome)/num_frames))
    return score

get_model_score_vmap = jax.vmap(get_model_score, in_axes = (0,None))


def weighted_rmse(A, F, W):
    return jnp.sqrt(jnp.sum(W * (A - F)**2) / jnp.sum(W))

def get_rg_distribution(arr):
    one_hot = jax.nn.one_hot(arr, num_classes=3)  # Shape: (N, T, 3)
    counts = jnp.sum(one_hot, axis=0)  # Shape: (T, 3)
    normalized = counts / jnp.sum(counts, axis=1, keepdims=True)  # Shape: (T, 3)
    return normalized


def create_log_frequency_heatmaps(jtap_metrics, targeted_analysis=False, bins=20, 
                                  weighted=False, cmap='viridis', cmap_reverse=False, 
                                  model_name='JTAP'):
    """
    Create log-frequency heatmaps comparing human and model decision patterns.
    
    Args:
        jtap_metrics: Dictionary mapping model_type -> DecisionMetrics
        targeted_analysis: If True, plot all three models in 3x2 grid; if False, plot only 'model' in 1x2 grid
        bins: Number of bins for the heatmap (default: 20)
        weighted: Whether to use correlation weights (default: False)
        cmap: Matplotlib colormap name (default: 'viridis')
        cmap_reverse: Whether to reverse the colormap (default: False)
        model_name: Name to display for the main model (default: 'JTAP')
    
    Returns:
        matplotlib.figure.Figure: The complete figure with all heatmaps
    """
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
        metrics = jtap_metrics[model_type]
        
        # Extract decision probabilities (P(Red or Green))
        human_decision_probs = metrics.human_decision_probs[metrics.valid_decision_mask]
        model_decision_probs = metrics.model_decision_probs[metrics.valid_decision_mask]
        
        # Extract conditional green probabilities (P(Green | Red or Green))
        human_green_given_decision = metrics.human_green_given_decision[metrics.valid_conditional_mask]
        model_green_given_decision = metrics.model_green_given_decision[metrics.valid_conditional_mask]
        decision_weights = metrics.decision_weights[metrics.valid_conditional_mask]
        
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
        green_corr = metrics.weighted_conditional_green_corr if weighted else metrics.conditional_green_corr
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
            ax2.set_title(f"$r_{{\\text{{wtd}}}} = {green_corr:.2f}$", fontsize=24, fontweight='bold', pad=10)
        else:
            # Labels for 1x2 layout
            ax1.set_xlabel(title, fontsize=24, fontweight='bold', labelpad=15)
            ax2.set_xlabel(title, fontsize=24, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Participants', fontsize=24, fontweight='bold', labelpad=15)
            ax1.text(0.01, 0.925, f"$r = {pressed_corr:.2f}$", fontsize=24, fontweight='bold', transform=ax1.transAxes)
            ax2.text(0.01, 0.925, f"$r_{{\\text{{wtd}}}} = {green_corr:.2f}$", fontsize=24, fontweight='bold', transform=ax2.transAxes)
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


def draw_trajectory_avoiding_dark_pixels(image, pdata, FPS, line_color=(0, 0, 0), 
                                         line_thickness=0.5, text_color=(0, 0, 0), 
                                         font_scale=1.0, font_thickness=0.5):
    upscale_factor = 4
    high_res_image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)

    world_to_image_scale = 40  # Adjust scale for the upscaled image
    sorted_frames = sorted(pdata.keys())

    for i in range(1, len(sorted_frames)):
        prev_frame = sorted_frames[i - 1]
        current_frame = sorted_frames[i]

        # Convert world coordinates to image coordinates for upscaled image
        x1, y1 = (
            (pdata[prev_frame]['x'] + 0.5) * world_to_image_scale, 
            800 - (pdata[prev_frame]['y'] + 0.5) * world_to_image_scale
        )
        x2, y2 = (
            (pdata[current_frame]['x'] + 0.5) * world_to_image_scale, 
            800 - (pdata[current_frame]['y'] + 0.5) * world_to_image_scale
        )

        # Draw line segment for the trajectory
        cv2.line(
            high_res_image, 
            (int(x1), int(y1)), 
            (int(x2), int(y2)), 
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
                    if not np.any(np.all(region < 27, axis=-1)):# and not np.any(np.all(region < 30, axis=-1)):
                        return nx, ny
        return x, y  # Default to the original position if no clear spot is found

    # Annotate seconds on the trajectory
    for frame in sorted_frames:
        if frame % FPS == 0:
            text = str(frame // FPS)
            x, y = (
                (pdata[frame]['x'] + 0.5) * world_to_image_scale, 
                800 - (pdata[frame]['y'] + 0.5) * world_to_image_scale
            )

            # Find a clear position to place the text
            x_clear, y_clear = find_clear_position(high_res_image, x, y - 15, text)

            # y_adjust = -50 if frame//FPS == 0 else 0
            y_adjust = 0

            # Annotate the frame number
            cv2.putText(
                high_res_image,
                text,  # Frame number as text
                (int(x_clear), int(y_clear - y_adjust)),  # Adjusted position
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  # Larger font scale for high resolution
                text_color,
                thickness=int(font_thickness * upscale_factor),  # Adjust thickness for high resolution
                lineType=cv2.LINE_AA
            )

    return high_res_image


from statsmodels.stats.weightstats import DescrStatsW

def weighted_corr(x, y, weights, epsilon = 1e-8):
    # Combine x and y into 2D array


    x_std = np.std(x)
    y_std = np.std(y)    
    if x_std == 0 and y_std == 0:
        return 1.0
    
    # if x_std == 0 or y_std == 0:
    #     return 0.0
    
    
    xy = np.vstack([x, y])
    
    # Create weighted stats object
    descstats = DescrStatsW(xy.T, weights=weights)
    
    # Get correlation matrix
    corr_matrix = descstats.corrcoef
    
    # Return correlation between x and y
    return corr_matrix[0,1]


def get_human_data(human_data_pkl_file, skip_t, global_trial_names):    
    # Load cleaned and arranged human data from the SQL database
    with open(human_data_pkl_file, "rb") as f:
        data = pickle.load(f)
        session_df = data["Session"]
        trial_df = data["Trial"]
        keystate_df = data["KeyState"]
        position_data = data["position_data"]
        occlusion_durations = data["occlusion_durations"]
        occlusion_frames_ = data["occlusion_frames"]

    occlusion_frames = {}
    for trial_name in global_trial_names:
        if trial_name in occlusion_frames_:
            occ_frames_ = occlusion_frames_[trial_name]
            occlusion_frames[trial_name] = [int(t/skip_t) for t in occ_frames_ if t % skip_t == 0]
        else:
            occlusion_frames[trial_name] = []
    reduced_keystate_df = keystate_df[keystate_df['frame'] % skip_t == 0]
    reduced_keystate_df = reduced_keystate_df[reduced_keystate_df['frame'] != 0]# NOTE: IGNORE T = 0

    reduced_position_data = {}
    for trial_name in global_trial_names:
        reduced_position_data[trial_name] = {k:v for k,v in position_data[trial_name].items() if k % skip_t == 0}

    HUMAN_stacked_key_presses = {}
    for trial_name in tqdm(global_trial_names, desc = "Calculating human key presses per trial"):
        reduced_keyframe_trial_df = reduced_keystate_df[reduced_keystate_df['global_trial_name'] == trial_name]
        num_frames = len(reduced_keyframe_trial_df['frame'].unique())
        all_keypresses = []
        for i in range(1, num_frames+1): # NOTE: IGNORE T = 0
            reduced_keyframe_trial_df_frame = reduced_keyframe_trial_df[reduced_keyframe_trial_df['frame'] == i*skip_t]
            green = jnp.array(list(reduced_keyframe_trial_df_frame['green']))
            red = jnp.array(list(reduced_keyframe_trial_df_frame['red']))
            uncertain = jnp.array(list(reduced_keyframe_trial_df_frame['uncertain']))
            keypress = 0*green + 1*red + 2*uncertain
            all_keypresses.append(keypress)
        all_keypresses = jnp.array(all_keypresses).T
        HUMAN_stacked_key_presses[trial_name] = all_keypresses

    HUMAN_stacked_key_dist = {k:get_rg_distribution(v) for k,v in HUMAN_stacked_key_presses.items()}
    HUMAN_stacked_key_SWITCHES = {k:jnp.sum(v[:,1:] != v[:,:-1], axis = 1) for k, v in HUMAN_stacked_key_presses.items()}


    HUMAN_stacked_scores = {}
    for trial_name, stacked_key_presses in HUMAN_stacked_key_presses.items():
        rg_outcome_idx = trial_df[trial_df['global_trial_name'] == trial_name]['rg_outcome_idx'].tolist()[0]
        scores = get_model_score_vmap(stacked_key_presses, rg_outcome_idx)
        HUMAN_stacked_scores[trial_name] = scores
    HUMAN_scores = {trial_name: jnp.mean(scores) for trial_name, scores in HUMAN_stacked_scores.items()}

    return session_df, trial_df, keystate_df, position_data, occlusion_durations, occlusion_frames,\
        HUMAN_stacked_key_presses, HUMAN_stacked_key_dist, HUMAN_scores, HUMAN_stacked_scores, HUMAN_stacked_key_SWITCHES

def get_decision_choice_values(ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, 
        ALL_stacked_key_dist_BASELINE_decayed, HUMAN_stacked_key_dist, HUMAN_stacked_key_presses, occlusion_frames = None):

    if occlusion_frames is None:
        keypress_dist_over_time_model = jnp.concatenate(list(ALL_stacked_key_dist.values()), axis=0)
        keypress_dist_over_time_frozen = jnp.concatenate(list(ALL_stacked_key_dist_BASELINE_frozen.values()), axis=0)
        keypress_dist_over_time_decayed = jnp.concatenate(list(ALL_stacked_key_dist_BASELINE_decayed.values()), axis=0)
        keypress_dist_over_time_HUMAN = jnp.concatenate(list(HUMAN_stacked_key_dist.values()), axis=0)
        ALL_human_bools = jnp.concatenate([jnp.sum(h_key_presses != 2, axis = 0) for h_key_presses in HUMAN_stacked_key_presses.values()], axis = 0)
    else:
        keypress_dist_over_time_model = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist.items()], axis = 0)
        keypress_dist_over_time_frozen = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist_BASELINE_frozen.items()], axis = 0)
        keypress_dist_over_time_decayed = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist_BASELINE_decayed.items()], axis = 0)
        keypress_dist_over_time_HUMAN = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in HUMAN_stacked_key_dist.items() if k in ALL_stacked_key_dist], axis = 0)
        ALL_human_bools = jnp.concatenate([jnp.sum(h_key_presses != 2, axis = 0)[jnp.array(occlusion_frames[k])] for k,h_key_presses in HUMAN_stacked_key_presses.items() if k in ALL_stacked_key_dist], axis = 0)

    # P(Green or Red)
    DECISION_DIST_PRESSED_BUTTON_model = jnp.sum(keypress_dist_over_time_model[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_frozen = jnp.sum(keypress_dist_over_time_frozen[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_decayed = jnp.sum(keypress_dist_over_time_decayed[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_HUMAN = jnp.sum(keypress_dist_over_time_HUMAN[:,:2], axis = 1)

    # P(Green | Green or Red)
    DECISION_DIST_CONDITIONAL_GREEN_model = keypress_dist_over_time_model[:,0]/DECISION_DIST_PRESSED_BUTTON_model
    DECISION_DIST_CONDITIONAL_GREEN_frozen = keypress_dist_over_time_frozen[:,0]/DECISION_DIST_PRESSED_BUTTON_frozen
    DECISION_DIST_CONDITIONAL_GREEN_decayed = keypress_dist_over_time_decayed[:,0]/DECISION_DIST_PRESSED_BUTTON_decayed
    DECISION_DIST_CONDITIONAL_GREEN_HUMAN = keypress_dist_over_time_HUMAN[:,0]/DECISION_DIST_PRESSED_BUTTON_HUMAN

    common_time_points_press_model = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_model), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
    common_time_points_press_frozen = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_frozen), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
    common_time_points_press_decayed = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_decayed), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
                                                    
    valid_model_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_model[common_time_points_press_model]
    valid_human_conditional_green_v_model = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_model]
    correl_weights_model = ALL_human_bools[common_time_points_press_model]
    weighted_model_green_conditional_pearsonr = weighted_corr(valid_model_conditional_green, valid_human_conditional_green_v_model, correl_weights_model)
    model_conditional_green_pearsonr = pearsonr(valid_model_conditional_green, valid_human_conditional_green_v_model)[0]                                               

    valid_frozen_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_frozen[common_time_points_press_frozen]
    valid_human_conditional_green_v_frozen = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_frozen]
    correl_weights_frozen = ALL_human_bools[common_time_points_press_frozen]
    weighted_frozen_green_conditional_pearsonr = weighted_corr(valid_frozen_conditional_green, valid_human_conditional_green_v_frozen, correl_weights_frozen)
    frozen_conditional_green_pearsonr = pearsonr(valid_frozen_conditional_green, valid_human_conditional_green_v_frozen)[0]

    valid_decayed_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_decayed[common_time_points_press_decayed]
    valid_human_conditional_green_v_decayed = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_decayed]
    correl_weights_decayed = ALL_human_bools[common_time_points_press_decayed]
    weighted_decayed_green_conditional_pearsonr = weighted_corr(valid_decayed_conditional_green, valid_human_conditional_green_v_decayed, correl_weights_decayed)
    decayed_conditional_green_pearsonr = pearsonr(valid_decayed_conditional_green, valid_human_conditional_green_v_decayed)[0]

    return keypress_dist_over_time_model, keypress_dist_over_time_frozen, keypress_dist_over_time_decayed, keypress_dist_over_time_HUMAN, \
        DECISION_DIST_PRESSED_BUTTON_model, DECISION_DIST_PRESSED_BUTTON_frozen, DECISION_DIST_PRESSED_BUTTON_decayed, DECISION_DIST_PRESSED_BUTTON_HUMAN, \
        model_conditional_green_pearsonr, frozen_conditional_green_pearsonr, decayed_conditional_green_pearsonr, \
        weighted_model_green_conditional_pearsonr, weighted_frozen_green_conditional_pearsonr, weighted_decayed_green_conditional_pearsonr, \
        correl_weights_model, correl_weights_frozen, correl_weights_decayed, \
        valid_model_conditional_green, valid_frozen_conditional_green, valid_decayed_conditional_green, \
        valid_human_conditional_green_v_model, valid_human_conditional_green_v_frozen, valid_human_conditional_green_v_decayed

# FUNCTIONS BELOW ARE THOSE THAT I REVIVED



def custom_rg_curve_marquee(rg_data, skip_t_, skip_t0=True, FPS=30):
    # Determine start index
    start_idx = 1 if skip_t0 else 0

    # Improve figure resolution
    fig, ax = plt.subplots(figsize=(15, 2.5), dpi=150)

    # Compute time values
    timevals = skip_t_ * (np.arange(0, rg_data.shape[0] - start_idx) / FPS)

    # Plot each line with improved styling
    ax.plot(timevals, rg_data[start_idx:, 0], color='green', label='Green', linewidth=3)
    ax.plot(timevals, rg_data[start_idx:, 1], color='red', label='Red', linewidth=3)
    ax.plot(timevals, rg_data[start_idx:, 2], color='blue', label='Uncertain', linewidth=3)

    # Add grid and labels
    ax.set_xlabel('Time (s)', fontsize=20, labelpad=0, rotation = 0)
    ax.set_ylabel('Proportion', fontsize=20, labelpad=5)
    # ax.set_title('Red-Green-Uncertain Probabilities Over Time', fontsize=16, pad=15)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=14, labelrotation = 0)
    ax.set_yticks(np.linspace(0, 1, 3))

    # Adjust legend
    legend = ax.legend(loc='upper center', fontsize=14, frameon=True, edgecolor='gray', ncol=3, bbox_to_anchor=(0.315, 1))
    legend.get_frame().set_facecolor('snow')  # Set background color


    # Add grid with subtle style
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optimize layout
    fig.tight_layout()

    return fig