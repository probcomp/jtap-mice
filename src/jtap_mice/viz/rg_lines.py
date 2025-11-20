from jtap_mice.evaluation import JTAPMice_Beliefs
from jtap_mice.utils import JTAPMiceStimulus
from jtap_mice.viz.figure_visuals import draw_stimulus_image
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Color palette for LEFT/RIGHT (not GREEN/RED/BLUE)
COLORS = {
    'left': '#3498db',         # Blue-ish for Left
    'right': '#e74c3c',        # Red-ish for Right
    'uncertain': '#888888',    # Gray for Uncertain, if present
    'left_light': '#b6dafa',   # Lighter blue for fills
    'right_light': '#f6b3aa',  # Lighter red for fills
    'grid': '#e0e0e0',
    'text': '#2c3e50',
    'axis': '#7f8c8d',
}

def jtap_plot_lr_lines(
    lr_beliefs, 
    include_baselines=True, 
    include_start_frame=False, 
    stimulus=None, 
    show="model",
    remove_legend=False,
    show_std_band=False,
    show_all_beliefs=False,
    return_fig=False,
    jtap_run_idx=None,
    plot_stat="mean",
    include_stimulus=False
):
    """
    Plot LEFT/RIGHT beliefs over time. If include_baselines is True,
    show all (model, frozen, decaying) side by side.
    Otherwise, show only 'show' (model/frozen/decaying).
    New option:
      - show_all_beliefs: If True and multiple runs, plot all lines (thin/semi-transparent).
      - show_std_band: If True, a std band will be rendered (mean/median).
      - plot_stat: "mean" or "median".
      - include_stimulus: If True and stimulus is provided, draw in plot layout.
    """
    if plot_stat not in {"mean", "median"}:
        raise ValueError("plot_stat must be 'mean' or 'median'.")

    assert isinstance(lr_beliefs, JTAPMice_Beliefs), "lr_beliefs must be a JTAPMice_Beliefs"
    assert isinstance(stimulus, JTAPMiceStimulus) if stimulus is not None else True, "stimulus must be a JTAPMiceStimulus"
    
    # Validate include_stimulus
    if include_stimulus and stimulus is None:
        warnings.warn("include_stimulus=True but no stimulus provided. Plotting without stimulus image.")
        include_stimulus = False

    # Setup
    num_jtap_runs = getattr(lr_beliefs, "num_jtap_runs", 1)
    is_multiple_runs = num_jtap_runs > 1

    y_label = "Weighted Beliefs"

    # Logic for jtap_run_idx
    jtap_run_mode = (jtap_run_idx is not None)
    if jtap_run_mode:
        assert is_multiple_runs, "jtap_run_idx only makes sense if there are multiple runs"
        if not (0 <= jtap_run_idx < num_jtap_runs):
            raise ValueError(f"jtap_run_idx {jtap_run_idx} is out of range (0 to {num_jtap_runs-1})")

    # Helper functions
    def get_belief_and_label(show_key):
        key = show_key.lower()
        if key == "model" or key == "jtap":
            arr = lr_beliefs.model_beliefs
            label = "Model"
        elif key == "frozen":
            arr = lr_beliefs.frozen_beliefs
            label = "Frozen"
        elif key == "decaying" or key == "decay":
            arr = lr_beliefs.decaying_beliefs
            label = "Decaying"
        else:
            raise ValueError(f"Unknown value to extract from JTAPMice_Beliefs: {show_key}")
        return arr, label

    def process_array_for_plotting(arr):
        stat_fn = np.mean if plot_stat == "mean" else np.median
        # shape (runs, T, 2) typically [left, right]; older shapes with a possible 3rd channel 'uncertain'
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

    # ---------------------------- Plotting section ----------------------------

    # Generate stimulus image if needed
    stimulus_image = None
    stimulus_array = None
    stimulus_aspect = None
    if include_stimulus and stimulus is not None:
        num_frames = len(stimulus.ground_truth_positions)
        stimulus_frame = num_frames - 1 if num_frames > 0 else 0
        stimulus_image = draw_stimulus_image(stimulus, frame=stimulus_frame, show_blue_dotted_ring=False)
        stimulus_array = np.array(stimulus_image)
        h, w = stimulus_array.shape[:2]
        stimulus_aspect = w / h

    # Panel arrangement setup
    if include_stimulus and stimulus_image is not None:
        # Always put stimulus ABOVE the line plots, full width
        if include_baselines:
            n_plot_rows = 3
            fig = plt.figure(figsize=(12, 3 * (n_plot_rows+1)))  # 1 for stimulus row
            # Gridspec: 4 rows = 1 for stimulus, 3 for plots
            gs = GridSpec(n_plot_rows + 1, 1, figure=fig, height_ratios=[0.5] + [1] * n_plot_rows, hspace=0.4)
            stimulus_ax = fig.add_subplot(gs[0, 0])
            ax0 = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[2, 0], sharex=ax0)
            ax2 = fig.add_subplot(gs[3, 0], sharex=ax0)
            axes = [ax0, ax1, ax2]
            belief_keys = [("model", "Model"), ("frozen", "Frozen"), ("decaying", "Decaying")]
        else:
            arr, label = get_belief_and_label(show)
            fig = plt.figure(figsize=(10, 6))
            # Two rows: stimulus, then main plot
            gs = GridSpec(2, 1, figure=fig, height_ratios=[0.7, 1.3], hspace=0.35)
            stimulus_ax = fig.add_subplot(gs[0, 0])
            axes = [fig.add_subplot(gs[1, 0])]
            belief_keys = [(show, label)]
    else:
        # No stimulus
        if include_baselines:
            nrows = 3
            fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
            if nrows == 1:
                axes = [axes]
            belief_keys = [("model", "Model"), ("frozen", "Frozen"), ("decaying", "Decaying")]
        else:
            arr, label = get_belief_and_label(show)
            fig, axes = plt.subplots(1, 1, figsize=(8, 4))
            axes = [axes]
            belief_keys = [(show, label)]
        stimulus_ax = None

    for i, (key, label) in enumerate(belief_keys):
        ax = axes[i]
        arr, _ = get_belief_and_label(key)
        plot_arr, std_arr = process_array_for_plotting(arr)

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
        n_channels = plot_arr.shape[1]
        # By design, first 2 channels are Left, Right. If 3rd exists, it's Uncertain.
        channel_labels = ["Left", "Right"] + (["Uncertain"] if n_channels == 3 else [])
        channel_colors = [COLORS['left'], COLORS['right']] + ([COLORS['uncertain']] if n_channels == 3 else [])
        channel_fills = [COLORS['left_light'], COLORS['right_light']] + ([COLORS['uncertain']] if n_channels == 3 else [])

        if (
            show_all_beliefs
            and not jtap_run_mode
            and is_multiple_runs
            and arr is not None
        ):
            arr_to_use = arr
            linewidth = 1
            all_alpha = 0.25
            stat_fn = np.mean if plot_stat == "mean" else np.median
            start_idx = 0 if include_start_frame else 1

            if arr_to_use.ndim != 3 or arr_to_use.shape[0] != num_jtap_runs:
                raise ValueError(f"Beliefs array for all lines mode has wrong shape: {arr_to_use.shape} (expected ({num_jtap_runs}, T, 2/3))")
            n_runs = arr_to_use.shape[0]
            for rr in range(n_runs):
                for ch, col in enumerate(channel_colors):
                    this_arr = arr_to_use[rr, :, ch]
                    ax.plot(time_values, this_arr[start_idx: start_idx+len(time_values)], color=col, alpha=all_alpha, linewidth=linewidth, zorder=1)
            # Plot stat (mean/median) line
            plot_arr_stat = stat_fn(arr_to_use[:, start_idx:start_idx+len(time_values), :], axis=0)
            for ch, col in enumerate(channel_colors):
                ax.plot(time_values, plot_arr_stat[:, ch], color=col, linewidth=1.5, alpha=0.9, label=channel_labels[ch], zorder=2)
            if show_std_band:
                std_arr_stat = np.std(arr_to_use[:, start_idx:start_idx+len(time_values), :], axis=0)
                for ch, fillcol in enumerate(channel_fills):
                    ax.fill_between(time_values, 
                        plot_arr_stat[:, ch] - std_arr_stat[:, ch], 
                        plot_arr_stat[:, ch] + std_arr_stat[:, ch], 
                        color=fillcol, alpha=0.22, zorder=0)
            ax.axhline(y=0, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            ax.axhline(y=1, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
        else:
            for ch, col in enumerate(channel_colors):
                ax.plot(time_values, plot_arr[:, ch], label=channel_labels[ch], color=col, linewidth=1.5, alpha=0.9)
            if show_std_band and std_arr is not None and not jtap_run_mode:
                for ch, fillcol in enumerate(channel_fills):
                    ax.fill_between(time_values, 
                                   plot_arr[:, ch] - std_arr[:, ch], 
                                   plot_arr[:, ch] + std_arr[:, ch], 
                                   color=fillcol, alpha=0.22, zorder=0)
            ax.axhline(y=0, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            ax.axhline(y=1, color=COLORS['axis'], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

        if stimulus is not None:
            ax.set_xlabel("Time (s)", fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)
        else:
            ax.set_xlabel("Time (frames)", fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)
        ax.set_ylabel(y_label, fontsize=15, fontweight='medium', color=COLORS['text'], labelpad=8)

        # Title logic
        run_title_str = ""
        stat_descr = "(Mean" if plot_stat == "mean" else "(Median"
        if jtap_run_idx is not None:
            run_title_str = f" (Run {jtap_run_idx})"
        elif is_multiple_runs:
            run_title_str = f" {stat_descr} across {num_jtap_runs} runs)"
        title = f"{label}{run_title_str}"
        ax.set_title(title, fontsize=17, fontweight='semibold', color=COLORS['text'], pad=18)

        if not remove_legend:
            legend = ax.legend(
                fontsize=13, frameon=True, fancybox=True, shadow=True, 
                framealpha=0.95, edgecolor='none', facecolor='white',
                loc='upper center', bbox_to_anchor=(0.5, 1.425), ncol=len(channel_labels),
                borderpad=0.8, labelspacing=0.7, handlelength=2.5, columnspacing=1.3
            )
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
        
        ax.set_facecolor('white')

    # Add stimulus image if requested
    if include_stimulus and stimulus_image is not None and stimulus_ax is not None:
        h, w = stimulus_array.shape[:2]
        stimulus_ax.imshow(stimulus_array, aspect='equal')
        stimulus_ax.set_xlim(0, w)
        stimulus_ax.set_ylim(h, 0)
        stimulus_ax.set_xticks([])
        stimulus_ax.set_yticks([])
        stimulus_ax.set_xticklabels([])
        stimulus_ax.set_yticklabels([])
        for spine in stimulus_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        stimulus_name = stimulus.name if stimulus is not None else 'Stimulus'
        stimulus_ax.set_title(
            stimulus_name, 
            fontsize=18, 
            fontweight='bold',
            color='#2c3e50',
            pad=20,
            style='normal'
        )

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