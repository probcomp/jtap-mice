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

from jtap_mice.utils import discrete_obs_to_rgb, slice_pt
from jtap_mice.inference import JTAPMiceData
from jtap_mice.evaluation import compute_weight_component_correlations, get_lr_raw_beliefs

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import jax.numpy as jnp
from IPython.display import HTML


def compute_lr_beliefs_directly(jtap_inference, jtap_run_idx, pred_len, num_frames, num_particles):
    pred_lr = jtap_inference.prediction.lr
    weights = jtap_inference.weight_data.final_weights
    arr_idx = jtap_run_idx if pred_lr.shape[0] > 1 else 0
    pred_lr = pred_lr[arr_idx]
    w = weights[arr_idx]
    offset = pred_len if pred_len is not None else jtap_inference.prediction.x.shape[1]
    idx = min(offset, pred_lr.shape[1]) - 1
    coded_lr_hits = np.array(pred_lr[:, idx, :])
    w = np.array(w)
    normalized_probs = np.exp(w - np.max(w, axis=1, keepdims=True))
    normalized_probs = normalized_probs / np.sum(normalized_probs, axis=1, keepdims=True)
    left_probs = np.sum((coded_lr_hits == 0) * normalized_probs, axis=1)
    right_probs = np.sum((coded_lr_hits == 1) * normalized_probs, axis=1)
    uncertain_probs = np.sum((coded_lr_hits == 2) * normalized_probs, axis=1)
    lr_belief_probs = np.stack([left_probs, right_probs, uncertain_probs], axis=1)
    return lr_belief_probs

def stack_tracking_and_predictions(tracking_x, prediction_x, sample_idx, pred_steps_use):
    """ 
    Returns array (n_frames, pred_steps_use+1, len(sample_idx))
    """
    tracking_x = np.array(tracking_x)  # (n_frames, n_particles)
    prediction_x = np.array(prediction_x)  # (n_frames, pred_steps, n_particles)
    n_frames = tracking_x.shape[0]
    n_sel_particles = len(sample_idx)

    # Always pick the right particles up front
    tracking_x_sel = tracking_x[:, sample_idx]  # (n_frames, n_sel_particles)
    prediction_x_sel = prediction_x[:, :pred_steps_use, :][:, :, sample_idx]  # (n_frames, pred_steps_use, n_sel_particles)

    # step 0 is tracking x at time t
    # The rest are prediction_x
    pred_x_full = np.concatenate([tracking_x_sel[:, None, :], prediction_x_sel], axis=1)  # (n_frames, pred_steps_use+1, n_sel_particles)
    return pred_x_full

def animate_jtap_predictions(
    JTAPMICE_DATA, 
    pred_len=None, 
    jtap_run_idx=0, 
    image_scale=4, 
    max_particles_to_show=None, 
    stimulus=None,
    scn_dot_min_size=32, 
    scn_dot_max_size=210,
    pred_dot_min_size=25, 
    pred_dot_max_size=120,
    line_thick_min=2.7,
    line_thick_max=10.5,
    use_tqdm=True,
    fps=10
):
    """
    Visualize particles, predictions, and left/right beliefs as an animated matplotlib figure.
    The prediction panel always includes current tracking as step 0, predictions as steps 1...N.
    """
    from tqdm.notebook import tqdm

    def _squeeze(arr):
        arr = np.array(arr)
        return arr[jtap_run_idx] if arr.shape[0] > 1 else arr[0]
    inf = JTAPMICE_DATA.inference

    tracking_x = np.array(_squeeze(inf.tracking.x))
    tracking_y = np.array(_squeeze(inf.tracking.y)) if hasattr(inf.tracking, "y") else None
    weights = np.array(_squeeze(inf.weight_data.final_weights))
    prediction_x = np.array(_squeeze(inf.prediction.x))
    prediction_y = np.array(_squeeze(inf.prediction.y)) if hasattr(inf.prediction, "y") else None
    prediction_lr = np.array(_squeeze(inf.prediction.lr))

    num_frames, num_particles = tracking_x.shape
    pred_steps = prediction_x.shape[1]
    # Use at most pred_len steps; fallback to all available steps otherwise. Always show tracking as step 0.
    plot_pred_steps = pred_len if (pred_len is not None and pred_len <= pred_steps) else pred_steps
    plot_pred_steps_with_track = plot_pred_steps + 1

    # We'll determine the sample idx first.
    if (max_particles_to_show is not None) and (num_particles > max_particles_to_show):
        # Use only the top particles by initial (frame 0) final weights
        sample_idx = np.argsort(weights[0])[::-1][:max_particles_to_show]
    else:
        sample_idx = np.arange(num_particles)
    
    pred_x_full = stack_tracking_and_predictions(tracking_x, prediction_x, sample_idx, plot_pred_steps)
    # pred_x_full: (n_frames, plot_pred_steps+1, n_selected_particles)

    lr_belief_probs = compute_lr_beliefs_directly(
        JTAPMICE_DATA.inference,
        jtap_run_idx,
        plot_pred_steps,
        num_frames,
        num_particles
    )

    if stimulus is None:
        if hasattr(JTAPMICE_DATA, "stimulus"):
            stimulus = JTAPMICE_DATA.stimulus
        else:
            raise ValueError("Must provide the stimulus or ensure JTAPMICE_DATA.stimulus exists")
    if hasattr(stimulus, "rgb_video_highres"):
        rgb_vid = np.asarray(stimulus.rgb_video_highres)
    elif hasattr(stimulus, "rgb_video"):
        rgb_vid = np.asarray(stimulus.rgb_video)
    elif hasattr(stimulus, "discrete_obs"):
        rgb_vid = discrete_obs_to_rgb(stimulus.discrete_obs)
    else:
        raise ValueError("Could not find rgb frames in stimulus.")
    H, W = rgb_vid.shape[1:3]
    n_time = rgb_vid.shape[0]

    diameter = getattr(stimulus, "diameter")
    ball_radius_scene = diameter / 2.0

    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
    })

    fig = plt.figure(
        figsize=(image_scale * 7, image_scale * 4),
        facecolor='w'
    )
    fig.suptitle(
        "Image-conditioned Sequential Monte Carlo for Tracking and Prediction",
        fontsize=32,
        fontweight='bold',
        y=0.99
    )

    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2.5, 2], width_ratios=[4, 1, 0.01, 0.1])
    ax_pred = fig.add_subplot(gs[0:2, 0], facecolor='w')  # prediction (top left, takes 2 rows)
    ax_belief = fig.add_subplot(gs[0:2, 1], facecolor='w')  # LR bar (top right)
    ax_scene = fig.add_subplot(gs[2, 0], facecolor='w')    # Scene/image (bottom left)

    plt.subplots_adjust(left=0.16, right=0.99, top=0.90, bottom=0.01)

    ax_pred.set_title("Future Predicted Positions per Particle")
    ax_scene.set_title("Ball center position (Black dots sized by SMC weights) over RGB image")
    ax_belief.set_title("LR Belief (prob)")

    import matplotlib.patches as mpatches

    scn_dots = ax_scene.scatter(
        [], [], s=[], c='black', alpha=1,
        edgecolors='none', linewidths=0, zorder=3
    )

    img_artist = ax_scene.imshow(
        np.zeros_like(rgb_vid[0]), origin="upper", animated=True, zorder=0, extent=(0, W, H, 0)
    )

    obs_border_rect = mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=4, edgecolor='black', facecolor='none', zorder=2
    )
    ax_scene.add_patch(obs_border_rect)

    lines = []
    for _ in range(len(sample_idx)):
        line, = ax_pred.plot([], [], lw=line_thick_min, color='yellow', alpha=0.85, zorder=2)
        lines.append(line)
    pred_dots = ax_pred.scatter([], [], s=[], c='black', alpha=1.0, zorder=3, edgecolors='none', linewidths=0)
    bar_beliefs = ax_belief.bar(['Left', 'Right', 'Unc.'], [0, 0, 0], color=['#3399FF', '#FF9933', '#C0C0C0'])

    ax_pred.set_xlabel('X pos')
    ax_pred.set_ylabel('Prediction step (future, 0=current, 1=+1 frame, ...)')

    scene_dim = JTAPMICE_DATA.params.inference_input.scene_dim
    scene_dim = np.array(scene_dim)
    if scene_dim.size > 0:
        Wscene = float(np.ravel(scene_dim)[0])
    else:
        Wscene = 1.0

    ax_pred.set_xlim(0, Wscene)
    ax_pred.set_ylim(-0.5, plot_pred_steps_with_track - 0.5)
    ax_pred.set_xticks(np.arange(0, int(np.ceil(Wscene)) + 1, 1))
    ax_pred.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    y_tick_locs = np.arange(0, plot_pred_steps_with_track, 5)
    if len(y_tick_locs) == 0 or y_tick_locs[-1] != plot_pred_steps_with_track - 1:
        y_tick_locs = np.append(y_tick_locs, plot_pred_steps_with_track - 1)
    y_tick_locs = np.unique(y_tick_locs)
    ax_pred.set_yticks(y_tick_locs)
    ax_pred.set_yticklabels([f"+{yy}" for yy in y_tick_locs])
    ax_pred.grid(True, linestyle='--', alpha=0.3)

    ax_scene.set_axis_off()
    ax_scene.set_xlim(0, W)
    ax_scene.set_ylim(H, 0)
    ax_belief.set_ylim(0, 1.01)
    ax_belief.set_ylabel('Prob')
    ax_belief.set_xticks([0, 1, 2])
    ax_belief.set_xticklabels(['Left', 'Right', 'Unknown'])
    for spine in ["top", "right"]:
        ax_belief.spines[spine].set_visible(False)

    def normalize_weights(w):
        w = np.array(w)
        m = np.max(w)
        if m == 0:
            return np.ones_like(w)
        w = w - np.max(w)
        w = np.exp(w)
        return w / np.sum(w)

    def compute_dot_sizes(weights, min_size, max_size):
        wnorm = normalize_weights(weights)
        sizes = min_size + (max_size - min_size) * wnorm
        return sizes

    def compute_line_widths(weights, min_width, max_width):
        wnorm = normalize_weights(weights)
        ws = min_width + (max_width - min_width) * wnorm
        return ws

    anim_progress = {'bar': None, 'last_frame': -1}

    def animate(t):
        if use_tqdm:
            if anim_progress['bar'] is None:
                anim_progress['bar'] = tqdm(total=num_frames, desc="Rendering frames", leave=True)
                anim_progress['last_frame'] = -1
            if t > anim_progress['last_frame']:
                anim_progress['bar'].n = t + 1
                anim_progress['bar'].refresh()
                anim_progress['last_frame'] = t
            if (t + 1) == num_frames:
                anim_progress['bar'].close()

        img = rgb_vid[t]
        img_artist.set_data(img)
        img_artist.set_extent((0, W, H, 0))
        tx = tracking_x[t]
        tw = weights[t]
        s_idx = np.array(sample_idx)
        tx_samp = tx[s_idx]
        tw_samp = tw[s_idx]
        tx_samp_center = tx_samp + ball_radius_scene
        x_pix_center = tx_samp_center * (W / Wscene)
        y_pix = np.ones_like(x_pix_center) * (H / 2)

        sizes_scn = compute_dot_sizes(tw_samp, scn_dot_min_size, scn_dot_max_size)
        scn_dots.set_offsets(np.stack([x_pix_center, y_pix], axis=1))
        scn_dots.set_sizes(sizes_scn)

        # Now assemble predictions (including current tracking at step 0, then step 1 etc).
        # pred_x_full: (num_frames, plot_pred_steps_with_track, n_sel_particles)
        xs_pred_t = pred_x_full[t] + ball_radius_scene  # (plot_pred_steps_with_track, n_sel_particles)
        ys_pred_t = np.arange(plot_pred_steps_with_track)[:, None] * np.ones((1, len(s_idx)))  # (plot_pred_steps_with_track, n_sel_particles)

        # So that the dots always appear at proper y (fix bug for frame 0, dots offscreen)
        # For the scatter we want offset: (n_sel_particles, 2) - just step 0 (current time)
        pred_dots.set_offsets(np.stack([xs_pred_t[0], np.zeros_like(xs_pred_t[0])], axis=1))
        pred_dots.set_sizes(compute_dot_sizes(tw_samp, pred_dot_min_size, pred_dot_max_size))

        line_ws = compute_line_widths(tw_samp, line_thick_min, line_thick_max)
        for li, p_idx in enumerate(range(len(s_idx))):
            # For each displayed particle
            xs_particle = xs_pred_t[:, li]
            ys_particle = ys_pred_t[:, li]
            # Compute masking like original: prediction_lr only covers steps after 0 (i.e. pred only, not tracking)
            lrs = prediction_lr[t, :plot_pred_steps, s_idx[li]]
            mask = np.ones(len(xs_particle), dtype=bool)
            hits = np.where((lrs == 0) | (lrs == 1))[0]
            if len(hits) > 0:
                hit_idx = hits[0]
                # after tracking step (step 0), so mask after hit_idx+1+1
                mask[(hit_idx + 2):] = False
            lines[li].set_data(xs_particle[mask], ys_particle[mask])
            lines[li].set_linewidth(line_ws[li])
            lines[li].set_alpha(0.88)

        left_p, right_p, unc_p = lr_belief_probs[t]
        for i, bar in enumerate(bar_beliefs):
            bar.set_height([left_p, right_p, unc_p][i])
            if i == 0:
                bar.set_color("#0055CC" if left_p > right_p else "#3399FF")
            elif i == 1:
                bar.set_color("#FFA416" if right_p > left_p else "#FF9933")
            else:
                bar.set_color("#C0C0C0")
        ax_belief.set_ylim(0, 1.01)
        ax_belief.set_title("LR belief:\nL={:.2f}, R={:.2f}, U={:.2f}".format(left_p, right_p, unc_p), fontsize=26)
        for ax in [ax_pred, ax_scene]:
            ax.set_title(ax.get_title().split('\n')[0] + f"\nframe {t + 1}/{num_frames}", fontsize=26)
        return [img_artist, scn_dots] + lines + [pred_dots] + list(bar_beliefs) + [obs_border_rect]

    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=(1000/fps), blit=True, repeat=True
    )
    plt.close()
    return HTML(anim.to_jshtml())
