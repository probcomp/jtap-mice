from jtap_mice.evaluation import JTAPMice_Beliefs
from jtap_mice.utils import JTAPMiceStimulus
from jtap_mice.viz.figure_visuals import draw_stimulus_image
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

# Professional color palette
COLORS = {
    'green': 'green',        # Original green
    'red': 'red',            # Original red
    'blue': 'blue',          # Original blue
    'green_light': '#a8e6cf',  # Light green for fill
    'red_light': '#f8a5a5',    # Light red for fill
    'blue_light': '#a8d8f0',   # Light blue for fill
    'grid': '#e0e0e0',       # Light gray for grid
    'text': '#2c3e50',       # Dark blue-gray for text
    'axis': '#7f8c8d',       # Medium gray for axes
}

def jtap_plot_rg_lines(
    rg_values, 
    include_baselines=True, 
    include_start_frame=False, 
    stimulus=None, 
    show="model",
    include_human=False,
    remove_legend=False,
    show_std_band=False,
    show_all_beliefs=False,
    jtap_metrics=None,
    return_fig=False,
    jtap_run_idx=None,
    plot_stat="mean",
    include_stimulus=False
):
    """
    Plot RG beliefs over time. If include_baselines is True, show all 3 (model, frozen, decaying) side by side.
    Otherwise, show only the one specified by 'show' ('model', 'frozen', or 'decaying').
    
    New option:
      - show_all_beliefs: If True and data has multiple runs, plot all individual belief/decision lines (thin, semi-transparent). 
      - show_std_band: If True, a std band will be rendered around the mean/median, independently of show_all_beliefs.
      - plot_stat: Either "mean" (default) or "median". Determines what statistic is plotted (and for std band, std is used for both).
      - include_stimulus: If True and stimulus is provided, draw the stimulus image in the plot layout.
    """
    if plot_stat not in {"mean", "median"}:
        raise ValueError("plot_stat must be 'mean' or 'median'.")

    assert isinstance(rg_values, JTAPMice_Beliefs) or isinstance(rg_values, JTAP_Decisions), "rg_values must be a JTAPMice_Beliefs or JTAP_Decisions"
    assert isinstance(stimulus, JTAPMiceStimulus) if stimulus is not None else True, "stimulus must be a JTAPMiceStimulus"
    assert isinstance(jtap_metrics, JTAP_Metrics) if jtap_metrics is not None else True, "jtap_metrics must be a JTAP_Metrics"
    
    # Validate include_stimulus
    if include_stimulus and stimulus is None:
        warnings.warn("include_stimulus=True but no stimulus provided. Plotting without stimulus image.")
        include_stimulus = False

    # Setup
    num_jtap_runs = getattr(rg_values, "num_jtap_runs", 1)
    num_pseudo_participants = getattr(rg_values, "num_pseudo_participants", None)
    is_multiple_runs = num_jtap_runs > 1

    is_beliefs = isinstance(rg_values, JTAPMice_Beliefs)
    is_decisions = isinstance(rg_values, JTAP_Decisions)
    y_label = "Weighted Beliefs" if is_beliefs else "Proportions"

    # Handle human data conditions, with new logic for missing data + warning
    has_human_data = stimulus is not None and getattr(stimulus, "human_data", None) is not None
    show_human = include_human and is_decisions

    if include_human and is_decisions and not has_human_data:
        warnings.warn("include_human=True but no human data available in stimulus. Plotting without human data.")
        show_human = False

    # Don't compute metrics if there is no human data!
    if jtap_metrics is not None and not has_human_data:
        jtap_metrics = None
    show_rmse = jtap_metrics is not None and is_decisions

    # Logic for jtap_run_idx validity/usage
    jtap_run_mode = ((is_beliefs or is_decisions) and jtap_run_idx is not None)
    if jtap_run_mode:
        assert is_multiple_runs, "jtap_run_idx only makes sense if there are multiple runs"
        if not (0 <= jtap_run_idx < num_jtap_runs):
            raise ValueError(f"jtap_run_idx {jtap_run_idx} is out of range (0 to {num_jtap_runs-1})")

    # Helper functions
    def get_belief_and_label(show_key):
        key = show_key.lower()
        if key == "model" or key == "jtap":
            arr = rg_values.model_beliefs if is_beliefs else rg_values.model_output
            label = "Model"
            if is_decisions and jtap_run_idx is not None:
                arr = rg_values.model_keypresses
        elif key == "frozen":
            arr = rg_values.frozen_beliefs if is_beliefs else rg_values.frozen_output
            label = "Frozen"
            if is_decisions and jtap_run_idx is not None:
                arr = rg_values.frozen_keypresses
        elif key == "decaying" or key == "decay":
            arr = rg_values.decaying_beliefs if is_beliefs else rg_values.decaying_output
            label = "Decaying"
            if is_decisions and jtap_run_idx is not None:
                arr = rg_values.decaying_keypresses
        else:
            raise ValueError(f"Unknown value to extract from JTAPMice_Beliefs: {show_key}")
        return arr, label

    def get_rmse_values(show_key):
        if not show_rmse:
            return None, None
        key = show_key.lower()
        if key == "model" or key == "jtap":
            metrics = jtap_metrics.model_metrics
        elif key == "frozen":
            metrics = jtap_metrics.frozen_metrics
        elif key == "decaying" or key == "decay":
            metrics = jtap_metrics.decaying_metrics
        else:
            return None, None
        return getattr(metrics, "rmse_loss", None), getattr(metrics, "occ_rmse_loss", None)

    def process_array_for_plotting(arr, is_keypress=False):
        # For keypresses + single run selection
        if is_decisions and jtap_run_idx is not None and is_keypress:
            # arr can have shape (num_jtap_runs * pseudo, T) or (num_jtap_runs, T)
            if arr.ndim == 2 and arr.shape[0] == num_jtap_runs:
                single = arr[jtap_run_idx]  # shape (T,)
                # For single run without pseudo-participants, compute proportions directly
                T = single.shape[0]
                arr_plot = np.zeros((T, 3))
                arr_plot[:, 0] = (single == 0).astype(float)
                arr_plot[:, 1] = (single == 1).astype(float)
                arr_plot[:, 2] = ((single != 0) & (single != 1)).astype(float)
                return arr_plot, None
            elif arr.ndim == 2:  # (num_jtap_runs * pseudo, T)
                pseudo_per_run = arr.shape[0] // num_jtap_runs
                arr_reshaped = arr.reshape(num_jtap_runs, pseudo_per_run, arr.shape[1])  # (num_jtap_runs, pseudo, T)
                keypresses_for_run = arr_reshaped[jtap_run_idx]  # (pseudo, T)
                # Compute proportions across pseudo-participants for each timestep
                T = keypresses_for_run.shape[1]
                arr_plot = np.zeros((T, 3))
                arr_plot[:, 0] = np.mean(keypresses_for_run == 0, axis=0)
                arr_plot[:, 1] = np.mean(keypresses_for_run == 1, axis=0)
                arr_plot[:, 2] = np.mean((keypresses_for_run != 0) & (keypresses_for_run != 1), axis=0)
                return arr_plot, None
            else:
                raise ValueError(f"JTAP_Decisions keypress array of shape {arr.shape} not understood for single run mode")

        # For JTAP_Decisions - mean/median and std across runs (model_output etc)
        if is_decisions:
            stat_fn = np.mean if plot_stat == "mean" else np.median
            if is_multiple_runs:
                # Defensive: check shape before access!
                # Expect arr shape (num_jtap_runs * pseudo, T, 3)
                if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 3:
                    arr = arr[:,0,:]
                    arr = arr.reshape(num_jtap_runs, -1, arr.shape[1])
                    arr_mean_pseudo = stat_fn(arr, axis=1)
                    mean_arr = stat_fn(arr_mean_pseudo, axis=0)
                    std_arr = np.std(arr_mean_pseudo, axis=0)
                    return mean_arr, std_arr
                elif arr.ndim == 3:
                    if arr.shape[0] % num_jtap_runs != 0:
                        raise ValueError(f"Expected arr.shape[0]={arr.shape[0]} to be divisible by num_jtap_runs={num_jtap_runs}")
                    pseudo_participants_per_run = arr.shape[0] // num_jtap_runs
                    T = arr.shape[1]
                    arr_reshaped = arr.reshape(num_jtap_runs, pseudo_participants_per_run, T, 3)
                    arr_mean_pseudo = stat_fn(arr_reshaped, axis=1)  # (num_jtap_runs, T, 3)
                    mean_arr = stat_fn(arr_mean_pseudo, axis=0)  # (T, 3)
                    std_arr = np.std(arr_mean_pseudo, axis=0)  # (T, 3)
                    return mean_arr, std_arr
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    if arr.shape[0] == num_jtap_runs:
                        mean_arr = stat_fn(arr, axis=0, keepdims=True)
                        std_arr = np.std(arr, axis=0, keepdims=True)
                        return mean_arr, std_arr
                    elif num_pseudo_participants is not None and arr.shape[0] == num_pseudo_participants:
                        if num_pseudo_participants % num_jtap_runs != 0:
                            raise ValueError(f"num_pseudo_participants={num_pseudo_participants} is not divisible by num_jtap_runs={num_jtap_runs}")
                        pseudo_participants_per_run = num_pseudo_participants // num_jtap_runs
                        arr_reshaped = arr.reshape(num_jtap_runs, pseudo_participants_per_run, 1, 3)
                        arr_mean_pseudo = stat_fn(arr_reshaped, axis=1)  # (num_jtap_runs, 1, 3)
                        mean_arr = stat_fn(arr_mean_pseudo, axis=0)  # (1, 3)
                        std_arr = np.std(arr_mean_pseudo, axis=0)  # (1, 3)
                        return mean_arr, std_arr
                    elif arr.shape[0] % num_jtap_runs == 0 and arr.shape[0] > num_jtap_runs:
                        pseudo_participants_per_run = arr.shape[0] // num_jtap_runs
                        arr_reshaped = arr.reshape(num_jtap_runs, pseudo_participants_per_run, 1, 3)
                        arr_mean_pseudo = stat_fn(arr_reshaped, axis=1)
                        mean_arr = stat_fn(arr_mean_pseudo, axis=0)
                        std_arr = np.std(arr_mean_pseudo, axis=0)
                        return mean_arr, std_arr
                    else:
                        return arr, None
                else:
                    raise ValueError(f"Unexpected JTAP_Decisions array shape {arr.shape} for plotting in mean/median mode")
            else:
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim == 2:
                    pass
                else:
                    raise ValueError(f"JTAP_Decisions array shape {arr.shape} not supported for plotting (single run)")
                return arr, None

        # For JTAPMice_Beliefs
        if is_beliefs:
            stat_fn = np.mean if plot_stat == "mean" else np.median
            if jtap_run_mode:
                if arr.shape[0] != num_jtap_runs:
                    raise ValueError(f"Expected {num_jtap_runs} runs, got {arr.shape[0]} in beliefs array")
                arr = arr[jtap_run_idx]
                return arr, None
            else:
                if is_multiple_runs:
                    mean_arr = stat_fn(arr, axis=0)
                    std_arr = np.std(arr, axis=0)
                    return mean_arr, std_arr
                else:
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    return arr, None
        raise ValueError("Could not interpret array for plotting.")

    # ---------------------------- Plotting section ----------------------------

    # Generate stimulus image if needed
    stimulus_image = None
    stimulus_array = None
    stimulus_aspect = None
    if include_stimulus and stimulus is not None:
        # Get the last frame to show the full trajectory
        num_frames = len(stimulus.ground_truth_positions)
        stimulus_frame = num_frames - 1 if num_frames > 0 else 0
        stimulus_image = draw_stimulus_image(stimulus, frame=stimulus_frame, show_blue_dotted_ring=False)
        # Convert PIL image to numpy array
        stimulus_array = np.array(stimulus_image)
        # Calculate aspect ratio for proper display
        h, w = stimulus_array.shape[:2]
        stimulus_aspect = w / h

    # Panel arrangement setup with GridSpec for flexible layout
    if include_stimulus and stimulus_image is not None:
        # Use GridSpec for more flexible layout when stimulus is included
        # Stimulus goes on the left, plots on the right
        if include_baselines:
            if show_human:
                # 2x3 grid: 2 rows for plots, 3 cols (1 for stimulus, 2 for plots)
                # Increase height to prevent overlapping
                fig = plt.figure(figsize=(20, 10))
                # Adjust width ratios to accommodate stimulus on left with proper aspect
                stimulus_width = 0.6 if stimulus_aspect is not None else 0.6
                gs = GridSpec(2, 3, figure=fig, width_ratios=[stimulus_width, 1, 1], hspace=0.5, wspace=0.4)
                # Stimulus axis spans both rows on the left
                stimulus_ax = fig.add_subplot(gs[:, 0])
                # Create axes with sharey for consistency
                ax00 = fig.add_subplot(gs[0, 1])
                ax01 = fig.add_subplot(gs[0, 2], sharey=ax00)
                ax10 = fig.add_subplot(gs[1, 1], sharex=ax00)
                ax11 = fig.add_subplot(gs[1, 2], sharex=ax01, sharey=ax01)
                axes = np.array([[ax00, ax01], [ax10, ax11]])
                belief_keys = [
                    ("human", "Human"),
                    ("model", "Model"),
                    ("frozen", "Frozen"),
                    ("decaying", "Decaying")
                ]
            else:
                # 3x2 grid: 3 rows for plots, 2 cols (1 for stimulus, 1 for plots)
                # Increase height to prevent overlapping
                fig = plt.figure(figsize=(14, 3 * 4))
                stimulus_width = 0.6 if stimulus_aspect is not None else 0.6
                gs = GridSpec(3, 2, figure=fig, width_ratios=[stimulus_width, 1], hspace=0.5, wspace=0.4)
                # Stimulus axis spans all rows on the left
                stimulus_ax = fig.add_subplot(gs[:, 0])
                # Create axes with sharex
                ax0 = fig.add_subplot(gs[0, 1])
                ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)
                ax2 = fig.add_subplot(gs[2, 1], sharex=ax0)
                axes = [ax0, ax1, ax2]
                belief_keys = [("model", "Model"), ("frozen", "Frozen"), ("decaying", "Decaying")]
        else:
            arr, label = get_belief_and_label(show)
            if show_human:
                # 1x3 grid: 1 row, 3 cols (1 for stimulus, 2 for plots)
                fig = plt.figure(figsize=(20, 4))
                stimulus_width = 0.6 if stimulus_aspect is not None else 0.6
                gs = GridSpec(1, 3, figure=fig, width_ratios=[stimulus_width, 1, 1], wspace=0.4)
                stimulus_ax = fig.add_subplot(gs[0, 0])
                ax0 = fig.add_subplot(gs[0, 1])
                ax1 = fig.add_subplot(gs[0, 2], sharey=ax0)
                axes = [ax0, ax1]
                belief_keys = [("human", "Human"), (show, label)]
            else:
                # 1x2 grid: 1 row, 2 cols (1 for stimulus, 1 for plot)
                fig = plt.figure(figsize=(14, 4))
                stimulus_width = 0.6 if stimulus_aspect is not None else 0.6
                gs = GridSpec(1, 2, figure=fig, width_ratios=[stimulus_width, 1], wspace=0.4)
                stimulus_ax = fig.add_subplot(gs[0, 0])
                axes = [fig.add_subplot(gs[0, 1])]
                belief_keys = [(show, label)]
    else:
        # Original layout without stimulus
        if include_baselines:
            if show_human:
                fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=True)
                belief_keys = [
                    ("human", "Human"),
                    ("model", "Model"),
                    ("frozen", "Frozen"),
                    ("decaying", "Decaying")
                ]
            else:
                nrows = 3
                fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
                if nrows == 1:
                    axes = [axes]
                belief_keys = [("model", "Model"), ("frozen", "Frozen"), ("decaying", "Decaying")]
        else:
            arr, label = get_belief_and_label(show)
            if show_human:
                belief_keys = [("human", "Human"), (show, label)]
                fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
            else:
                belief_keys = [(show, label)]
                fig, axes = plt.subplots(1, 1, figsize=(8, 4))
                axes = [axes]
        stimulus_ax = None

    for i, (key, label) in enumerate(belief_keys):
        if include_baselines and show_human:
            row = i // 2
            col = i % 2
            ax = axes[row, col]
        else:
            ax = axes[i]

        if key == "human":
            arr = stimulus.human_data.human_output
            plot_arr, std_arr = arr, None
        else:
            arr, _ = get_belief_and_label(key)
            use_keypress = (
                is_decisions and jtap_run_idx is not None and (
                    ((key.lower() == "model" or key.lower() == "jtap") and hasattr(rg_values, "model_keypresses")) or
                    (key.lower() == "frozen" and hasattr(rg_values, "frozen_keypresses")) or
                    (key.lower() in {"decaying", "decay"} and hasattr(rg_values, "decaying_keypresses"))
                )
            )
            plot_arr, std_arr = process_array_for_plotting(arr, is_keypress=use_keypress)

        if not include_start_frame:
            plot_arr = plot_arr[1:]
            if std_arr is not None:
                std_arr = std_arr[1:]

        time_values = np.arange(plot_arr.shape[0])

        if stimulus is not None:
            fps = stimulus.fps
            skip_t = stimulus.skip_t
            adjusted_fps = fps / skip_t
            time_values = time_values / adjusted_fps

            fully_occluded = getattr(stimulus, "fully_occluded_bool", None)
            partially_occluded = getattr(stimulus, "partially_occluded_bool", None)
            if not include_start_frame:
                if fully_occluded is not None:
                    fully_occluded = fully_occluded[1:]
                if partially_occluded is not None:
                    partially_occluded = partially_occluded[1:]
            if partially_occluded is not None and fully_occluded is not None:
                only_partial = np.logical_and(partially_occluded, np.logical_not(fully_occluded))
                in_occl = False
                start = None
                for j, val in enumerate(only_partial):
                    if val and not in_occl:
                        in_occl = True
                        start = j
                    if ((not val) or (j == len(only_partial) - 1)) and in_occl:
                        end = j if not val else j + 1
                        time_step = time_values[1] - time_values[0]
                        ax.axvspan(
                            time_values[start] - (time_step/2), 
                            time_values[end-1] - (time_step/2) + (time_step if end < len(time_values) else 0), 
                            color='#d5d5d5', alpha=0.4, zorder=0
                        )
                        in_occl = False
            if fully_occluded is not None:
                in_occl = False
                start = None
                for j, val in enumerate(fully_occluded):
                    if val and not in_occl:
                        in_occl = True
                        start = j
                    if ((not val) or (j == len(fully_occluded) - 1)) and in_occl:
                        end = j if not val else j + 1
                        time_step = time_values[1] - time_values[0]
                        ax.axvspan(
                            time_values[start] - (time_step/2), 
                            time_values[end-1] - (time_step/2) + (time_step if end < len(time_values) else 0), 
                            color='#b0b0b0', alpha=0.35, zorder=1
                        )
                        in_occl = False

        # ----------- Main plot drawing logic ---------------

        if (
            show_all_beliefs
            and not jtap_run_mode
            and (is_beliefs or is_decisions)
            and not (key == "human")
            and is_multiple_runs
            and arr is not None
        ):
            arr_to_use = arr
            color_map = [COLORS['green'], COLORS['red'], COLORS['blue']]
            linewidth = 1
            all_alpha = 0.25
            stat_fn = np.mean if plot_stat == "mean" else np.median
            start_idx = 0 if include_start_frame else 1

            # Plot all belief/decision lines
            if is_beliefs:
                if arr_to_use.ndim != 3 or arr_to_use.shape[0] != num_jtap_runs:
                    raise ValueError(f"Beliefs array for all lines mode has wrong shape: {arr_to_use.shape} (expected ({num_jtap_runs}, T, 3))")
                n_runs = arr_to_use.shape[0]
                n_T = arr_to_use.shape[1]
                for rr in range(n_runs):
                    for ch, col in enumerate(color_map):
                        this_arr = arr_to_use[rr, :, ch]
                        ax.plot(time_values, this_arr[start_idx: start_idx+len(time_values)], color=col, alpha=all_alpha, linewidth=linewidth, zorder=1)
                # Plot stat (mean/median) line
                plot_arr_stat = stat_fn(arr_to_use[:, start_idx:start_idx+len(time_values), :], axis=0)
                for ch, col in enumerate(color_map):
                    ax.plot(time_values, plot_arr_stat[:, ch], color=col, linewidth=1.5, alpha=0.9, label=["Green", "Red", "Uncertain"][ch], zorder=2)
                # Plot std band if requested
                if show_std_band:
                    std_arr_stat = np.std(arr_to_use[:, start_idx:start_idx+len(time_values), :], axis=0)
                    ax.fill_between(time_values, 
                        plot_arr_stat[:, 0] - std_arr_stat[:, 0], 
                        plot_arr_stat[:, 0] + std_arr_stat[:, 0], 
                        color=COLORS['green'], alpha=0.25, zorder=0)
                    ax.fill_between(time_values, 
                        plot_arr_stat[:, 1] - std_arr_stat[:, 1], 
                        plot_arr_stat[:, 1] + std_arr_stat[:, 1], 
                        color=COLORS['red'], alpha=0.25, zorder=0)
                    # Uncomment to show Uncertain std band
                    # ax.fill_between(time_values, 
                    #     plot_arr_stat[:, 2] - std_arr_stat[:, 2], 
                    #     plot_arr_stat[:, 2] + std_arr_stat[:, 2], 
                    #     color="blue", alpha=0.2)
                # Draw dotted lines at y=0 and y=1
                ax.axhline(y=0, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
                ax.axhline(y=1, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            elif is_decisions:
                # shape inference
                if arr_to_use.ndim == 3:
                    total_0 = arr_to_use.shape[0]
                    n_T = arr_to_use.shape[1]
                    if total_0 == num_jtap_runs:
                        n_runs = num_jtap_runs
                        arr_for_lines = arr_to_use
                    elif num_pseudo_participants is not None and total_0 == num_pseudo_participants:
                        n_runs = 1
                        arr_for_lines = arr_to_use[np.newaxis, ...]
                    elif num_pseudo_participants is not None and total_0 == num_jtap_runs * (num_pseudo_participants // num_jtap_runs):
                        pseudo_per_run = num_pseudo_participants // num_jtap_runs
                        arr_for_lines = arr_to_use.reshape(num_jtap_runs, pseudo_per_run, n_T, 3)
                        n_runs = num_jtap_runs
                    elif total_0 % num_jtap_runs == 0 and total_0 > num_jtap_runs:
                        pseudo_per_run = total_0 // num_jtap_runs
                        arr_for_lines = arr_to_use.reshape(num_jtap_runs, pseudo_per_run, n_T, 3)
                        n_runs = num_jtap_runs
                    else:
                        raise ValueError(f"Decisions array shape for all lines not understood: {arr_to_use.shape}")
                    if arr_for_lines.ndim == 4:
                        # (num_jtap_runs, pseudo, T, 3)
                        for rr in range(arr_for_lines.shape[0]):
                            for pp in range(arr_for_lines.shape[1]):
                                this_arr = arr_for_lines[rr, pp, :, :]
                                for ch, col in enumerate(color_map):
                                    ax.plot(time_values, this_arr[start_idx: start_idx+len(time_values), ch],
                                            color=col, alpha=all_alpha, linewidth=linewidth, zorder=1)
                        stat_arr = stat_fn(arr_for_lines[:, :, start_idx:start_idx+len(time_values), :], axis=(0,1))
                        for ch, col in enumerate(color_map):
                            ax.plot(time_values, stat_arr[:, ch], color=col, linewidth=1.5, alpha=0.9, label=["Green", "Red", "Uncertain"][ch], zorder=2)
                        if show_std_band:
                            std_arr_stat = np.std(arr_for_lines[:, :, start_idx:start_idx+len(time_values), :], axis=(0,1))
                            ax.fill_between(time_values, 
                                stat_arr[:, 0] - std_arr_stat[:, 0], 
                                stat_arr[:, 0] + std_arr_stat[:, 0], 
                                color=COLORS['green'], alpha=0.25, zorder=0)
                            ax.fill_between(time_values, 
                                stat_arr[:, 1] - std_arr_stat[:, 1], 
                                stat_arr[:, 1] + std_arr_stat[:, 1], 
                                color=COLORS['red'], alpha=0.25, zorder=0)
                            # Uncomment to show Uncertain std band
                            # ax.fill_between(time_values, 
                            #     stat_arr[:, 2] - std_arr_stat[:, 2], 
                            #     stat_arr[:, 2] + std_arr_stat[:, 2], 
                            #     color="blue", alpha=0.2)
                    elif arr_for_lines.ndim == 3:
                        for rr in range(arr_for_lines.shape[0]):
                            for ch, col in enumerate(color_map):
                                this_arr = arr_for_lines[rr, :, ch]
                                ax.plot(time_values, this_arr[start_idx: start_idx+len(time_values)], color=col, alpha=all_alpha, linewidth=linewidth, zorder=1)
                        stat_arr = stat_fn(arr_for_lines[:, start_idx:start_idx+len(time_values), :], axis=0)
                        for ch, col in enumerate(color_map):
                            ax.plot(time_values, stat_arr[:, ch], color=col, linewidth=1.5, alpha=0.9, label=["Green", "Red", "Uncertain"][ch], zorder=2)
                        if show_std_band:
                            std_arr_stat = np.std(arr_for_lines[:, start_idx:start_idx+len(time_values), :], axis=0)
                            ax.fill_between(time_values, 
                                stat_arr[:, 0] - std_arr_stat[:, 0], 
                                stat_arr[:, 0] + std_arr_stat[:, 0], 
                                color=COLORS['green'], alpha=0.25, zorder=0)
                            ax.fill_between(time_values, 
                                stat_arr[:, 1] - std_arr_stat[:, 1], 
                                stat_arr[:, 1] + std_arr_stat[:, 1], 
                                color=COLORS['red'], alpha=0.25, zorder=0)
                            # Uncomment to show Uncertain std band
                            # ax.fill_between(time_values, 
                            #     stat_arr[:, 2] - std_arr_stat[:, 2], 
                            #     stat_arr[:, 2] + std_arr_stat[:, 2], 
                            #     color="blue", alpha=0.2)
                    else:
                        raise ValueError("Decisions array must be 3D for show_all_beliefs mode.")
                else:
                    raise ValueError("Decisions array must be 3D for show_all_beliefs mode.")
            # Draw dotted lines at y=0 and y=1 for decisions too (always in all-lines mode)
            ax.axhline(y=0, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            ax.axhline(y=1, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
        else:
            ax.plot(time_values, plot_arr[:, 0], label="Green", color=COLORS['green'], linewidth=1.5, alpha=0.9)
            ax.plot(time_values, plot_arr[:, 1], label="Red", color=COLORS['red'], linewidth=1.5, alpha=0.9)
            ax.plot(time_values, plot_arr[:, 2], label="Uncertain", color=COLORS['blue'], linewidth=1.5, alpha=0.9)
            # Plot std bands only if in mean/median mode and std_arr exists and desired by user
            if show_std_band and std_arr is not None and not jtap_run_mode:
                ax.fill_between(time_values, 
                               plot_arr[:, 0] - std_arr[:, 0], 
                               plot_arr[:, 0] + std_arr[:, 0], 
                               color=COLORS['green'], alpha=0.25, zorder=0)
                ax.fill_between(time_values, 
                               plot_arr[:, 1] - std_arr[:, 1], 
                               plot_arr[:, 1] + std_arr[:, 1], 
                               color=COLORS['red'], alpha=0.25, zorder=0)
                # Uncomment to show Uncertain std band
                # ax.fill_between(time_values, 
                #                plot_arr[:, 2] - std_arr[:, 2], 
                #                plot_arr[:, 2] + std_arr[:, 2], 
                #                color="blue", alpha=0.2)
            # Draw dotted lines at y=0 and y=1 for beliefs and for decisions
            if (is_beliefs or is_decisions):
                ax.axhline(y=0, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
                ax.axhline(y=1, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

        if stimulus is not None:
            ax.set_xlabel("Time (s)", fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)
        else:
            ax.set_xlabel("Time (frames)", fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)
        ax.set_ylabel(y_label, fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)

        # Title logic
        if key == "human":
            title = label
        else:
            run_title_str = ""
            stat_descr = "(Mean" if plot_stat == "mean" else "(Median"
            if is_beliefs:
                if jtap_run_idx is not None:
                    run_title_str = f" (Run {jtap_run_idx})"
                elif is_multiple_runs:
                    run_title_str = f" {stat_descr} across {num_jtap_runs} runs)"
            elif is_decisions:
                if jtap_run_idx is not None:
                    if num_pseudo_participants is not None:
                        pseudo_per_run = num_pseudo_participants // num_jtap_runs
                        run_title_str = f" (Run {jtap_run_idx}, {pseudo_per_run} pseudo-participants)"
                    else:
                        run_title_str = f" (Run {jtap_run_idx})"
                elif is_multiple_runs:
                    run_title_str = f" {stat_descr} across {num_jtap_runs} runs)"
            title = f"{label}{run_title_str}"
            rmse, occ_rmse = get_rmse_values(key)
            if rmse is not None:
                if occ_rmse is not None:
                    title += f"\nRMSE: {rmse:.3f}, Occ RMSE: {occ_rmse:.3f}"
                else:
                    title += f"\nRMSE: {rmse:.3f}"
        ax.set_title(title, fontsize=17, fontweight='semibold', color=COLORS['text'], pad=18)

        if not remove_legend:
            legend = ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True, 
                             framealpha=0.95, edgecolor='none', facecolor='white',
                             loc='best', borderpad=0.8, labelspacing=0.7)
            # Make legend text slightly bolder
            for text in legend.get_texts():
                text.set_fontweight('medium')
        
        # Improve tick styling
        ax.tick_params(axis='both', which='major', labelsize=13, colors=COLORS['text'], 
                      width=0.8, length=4, pad=5)
        ax.tick_params(axis='both', which='minor', labelsize=11, colors=COLORS['text'])
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_color(COLORS['axis'])
            spine.set_linewidth(0.8)
            spine.set_alpha(0.7)
        
        # Add subtle grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['grid'], alpha=0.5, zorder=0)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['grid'], alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # Set face color to white for clean look
        ax.set_facecolor('white')

    # Add stimulus image if requested
    if include_stimulus and stimulus_image is not None and stimulus_ax is not None:
        # Maintain aspect ratio - use 'equal' to preserve original aspect ratio
        h, w = stimulus_array.shape[:2]
        stimulus_ax.imshow(stimulus_array, aspect='equal')
        # Set axis limits to match image dimensions (flip y-axis for image coordinates)
        stimulus_ax.set_xlim(0, w)
        stimulus_ax.set_ylim(h, 0)
        
        # Turn off ticks and labels but keep spines for border
        stimulus_ax.set_xticks([])
        stimulus_ax.set_yticks([])
        stimulus_ax.set_xticklabels([])
        stimulus_ax.set_yticklabels([])
        
        # Add thin black border around the stimulus
        for spine in stimulus_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        # Get stimulus name for title with improved styling
        stimulus_name = stimulus.name if stimulus is not None else 'Stimulus'
        stimulus_ax.set_title(
            stimulus_name, 
            fontsize=18, 
            fontweight='bold',
            color='#2c3e50',  # Dark blue-gray color
            pad=20,
            style='normal'
        )

    # Set figure background to white for clean look
    if include_stimulus and stimulus_image is not None:
        fig.patch.set_facecolor('white')
    else:
        fig.patch.set_facecolor('white')

    # Suppress the UserWarning about tight_layout and incompatible Axes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        return None