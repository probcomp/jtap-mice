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
from jtap_mice.evaluation import get_lr_raw_beliefs
from jtap_mice.inference import JTAPMiceData
from jtap_mice.utils import discrete_obs_to_rgb, d2r, r2d

def interpretable_belief_viz(JTAPMice_data, high_res_video, prediction_t_offset=5, video_offset=(0, 0), viz_key=jax.random.PRNGKey(0), 
                  min_dot_alpha=0.2, min_line_alpha=0.04,
                  ax_num=None, timeframe=None, high_res_multiplier=5, diameter=1.0):

    assert isinstance(JTAPMice_data, JTAPMiceData), "JTAPMice_data must be of type JTAPMiceData"

    inference_input = JTAPMice_data.params.inference_input

    def generate_samples(key, n_samples, support, logprobs):
        keys = jax.random.split(key, n_samples)
        sampled_idxs = jax.vmap(jax.random.categorical, in_axes = (0, None))(keys, logprobs)
        return support[sampled_idxs]

    generate_samples_vmap = jax.vmap(generate_samples, in_axes = (0, None, None, None))

    max_line_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_line_alpha = min_line_alpha

    max_dot_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_dot_alpha = min_dot_alpha

    num_inference_steps = JTAPMice_data.inference.weight_data.final_weights.shape[0]
    num_prediction_steps = JTAPMice_data.params.max_prediction_steps
    max_inference_T = num_inference_steps - 1
    maxt = high_res_video.shape[0]
    n_particles = JTAPMice_data.params.num_particles
    normalized_weights = jnp.exp(JTAPMice_data.inference.weight_data.final_weights - logsumexp(JTAPMice_data.inference.weight_data.final_weights, axis = 1, keepdims = True)) # T by N_PARTICLES

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


    inf_x_points = JTAPMice_data.inference.tracking.x # T by N_PARTICLES
    inf_y_points = JTAPMice_data.inference.tracking.y # T by N_PARTICLES
    pred_x_lines = jnp.concatenate([JTAPMice_data.inference.tracking.x[:,None,:], JTAPMice_data.inference.prediction.x], axis = 1) # T by T_pred+1 by N_PARTICLES
    pred_y_lines = jnp.concatenate([JTAPMice_data.inference.tracking.y[:,None,:], JTAPMice_data.inference.prediction.y], axis = 1) # T by T_pred+1 by N_PARTICLES
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
    scale_multiplier = 1 / (JTAPMice_data.params.image_discretization)*high_res_multiplier
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
    lr_data = get_lr_raw_beliefs(JTAPMice_data, prediction_t_offset)
    bars = ax2.bar(range(3), lr_data[0], color = ['green', 'red', 'blue'])

    # Set up the axis limits
    # Set title and x-ticks
    ax2.set_title('Left or Right: Which will it hit next?', fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Left', 'Right', 'Uncertain'])
    ax2.set_ylim(0, 1)

    # AX3
    num_bins = 90
    n_samples = 10000
    max_count = 10
    viz_key, dir_key = jax.random.split(viz_key, 2)
    dir_keys = jax.random.split(dir_key, num_inference_steps)
    sampled_dir = generate_samples_vmap(dir_keys, n_samples, JTAPMice_data.inference.tracking.direction, JTAPMice_data.inference.weight_data.final_weights)
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
    sampled_speed = generate_samples_vmap(speed_keys, n_samples, JTAPMice_data.inference.tracking.speed, JTAPMice_data.inference.weight_data.final_weights)
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


        for bar, height in zip(bars, lr_data[idx]):
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

import cv2
from PIL import Image
import numpy as np

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
    show_blue_dotted_ring=True,
    do_not_draw_seconds=True,
    offset_line_proportion_ball=0.25,
    gradient_trajectory=True,
    fade_older=False,
    # --- New controls for user questions below:
    arrow_length_px=6,        # Controls arrow length in pixels (before upscale)
    arrow_thickness_px=None,   # Controls arrow thickness (None = automatic, else int)
    gradient_extent=0.25,       # Controls how "extreme" the color gradient is, 0=constant color, 1=full spectrum
):
    """
    Visualizes stimulus trajectory with customizable offset, color gradient, and arrow marker properties.

    New args:
    - arrow_length_px: Arrow base length (int, before upscale; e.g. 10 = original, 20 = longer, 5 = shorter).
    - arrow_thickness_px: Arrow line thickness (int, before upscale; default auto).
    - gradient_extent: Controls how intense the color gradient is (0.0 = uniform color, 1.0 = full HSV sweep).
    """

    # ---- Setups ----
    ground_truth_positions = jtap_stimulus.ground_truth_positions
    discrete_obs = jtap_stimulus.discrete_obs
    stimulus_fps = jtap_stimulus.fps
    skip_t = jtap_stimulus.skip_t

    if FPS is None:
        FPS = stimulus_fps

    effective_fps = stimulus_fps / skip_t

    frame_discrete_obs = discrete_obs[frame:frame + 1]
    rgb_frame = discrete_obs_to_rgb(frame_discrete_obs)[0]

    upscale_factor = 4
    original_height, original_width = rgb_frame.shape[:2]
    upscale_height, upscale_width = original_height * upscale_factor, original_width * upscale_factor
    high_res_image = cv2.resize(rgb_frame, (upscale_width, upscale_height), interpolation=cv2.INTER_NEAREST)

    diameter = getattr(jtap_stimulus, "diameter", 1)
    pixel_density = getattr(jtap_stimulus, "pixel_density", 10)
    image_height = original_height
    offset_ballunits = diameter * offset_line_proportion_ball
    offset_pixels = offset_ballunits * pixel_density * upscale_factor

    def xy_to_uv(x, y):
        return (
            (x + (0.5 * diameter)) * pixel_density,
            image_height - ((y + (0.5 * diameter)) * pixel_density)
        )

    # -- Prepare trajectory positions (with offset) and directions --
    positions_xy = np.array(ground_truth_positions)
    n_traj = len(positions_xy)
    xs, ys = positions_xy[:,0], positions_xy[:,1]

    # Compute direction: +1 (right), -1 (left), 0 (constant)
    dx = np.diff(xs)
    direction = np.zeros_like(xs)
    direction[1:] = np.sign(dx)
    # Set the first direction to the first nonzero, if needed
    for i, d in enumerate(direction[1:], 1):
        if d != 0:
            direction[0] = d
            break

    # -- Get all points for original and offset line --
    offset_sign = 1
    points_main = []
    points_offset = []
    for x, y in positions_xy:
        xuv, yuv = xy_to_uv(x, y)
        points_main.append((xuv * upscale_factor, yuv * upscale_factor))
        points_offset.append((xuv * upscale_factor, (yuv + offset_sign * offset_pixels / upscale_factor) * upscale_factor))

    # -------------- Trajectory Color/Alpha Utilities with Modifiable Gradient ---------------
    def color_gradient(i, N):
        if not gradient_trajectory or gradient_extent <= 0:
            return line_color
        # Less extreme gradient: restrict hue swing to [0, 255 * gradient_extent]
        # gradient_extent=1.0: full HSV, 0.5: half of hue scale, 0=no grad.
        hue_swing = int(255 * float(gradient_extent))
        hue = int(hue_swing * i / max(1, N))
        sat = 255
        val = 255
        hsv = np.uint8([[[hue, sat, val]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        return tuple(int(x) for x in rgb)

    def alpha_for_segment(i, N):
        if not fade_older:
            return 1.0
        min_alpha = 0.25
        max_alpha = 1.0
        fade_frac = i / max(1, N)
        return min_alpha + (max_alpha - min_alpha) * fade_frac

    # ------ Find direction-flip indexes ------
    def get_direction_flip_indices(direction):
        flips = []
        for i in range(1, len(direction)):
            if direction[i-1] != 0 and direction[i] != 0 and direction[i] != direction[i-1]:
                flips.append(i)
        return flips
    flip_indices = get_direction_flip_indices(direction)

    # ------ Draw Trajectory ------
    for i in range(1, n_traj):
        color = color_gradient(i-1, n_traj-1)
        alpha = alpha_for_segment(i-1, n_traj-1)

        d = direction[i]
        d_prev = direction[i-1]
        line_on_offset = d != 0 and d > 0

        if line_on_offset:
            pt1 = points_offset[i-1]
            pt2 = points_offset[i]
        else:
            pt1 = points_main[i-1]
            pt2 = points_main[i]

        if alpha >= 0.999:
            cv2.line(
                high_res_image,
                (int(round(pt1[0])), int(round(pt1[1]))),
                (int(round(pt2[0])), int(round(pt2[1]))),
                color,
                int(line_thickness * upscale_factor),
                lineType=cv2.LINE_AA,
            )
        else:
            temp = high_res_image.copy()
            cv2.line(
                temp,
                (int(round(pt1[0])), int(round(pt1[1]))),
                (int(round(pt2[0])), int(round(pt2[1]))),
                color,
                int(line_thickness * upscale_factor),
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(temp, alpha, high_res_image, 1 - alpha, 0, dst=high_res_image)

        if i in flip_indices:
            if d_prev > 0:
                p_from = points_offset[i-1]
                p_to = points_main[i]
            else:
                p_from = points_main[i-1]
                p_to = points_offset[i]
            cv2.line(
                high_res_image,
                (int(round(p_from[0])), int(round(p_from[1]))),
                (int(round(p_to[0])), int(round(p_to[1]))),
                (64,64,64),
                int(line_thickness * upscale_factor) + 1,
                lineType=cv2.LINE_AA,
            )

    # ------ Draw markers/arrows on corresponding line ------
    def get_tick_points_and_colors():
        frames_per_second_annotation = effective_fps
        max_time_seconds = (n_traj-1) / effective_fps
        num_whole_seconds = int(np.floor(max_time_seconds)) + 1
        ticks = []
        for s in range(num_whole_seconds):
            target_frame = s * effective_fps
            frame_before = int(np.floor(target_frame))
            frame_after = int(np.ceil(target_frame))
            if frame_before == frame_after and frame_before < n_traj:
                t = 0
                use_frame = frame_before
            elif frame_before < n_traj and frame_after < n_traj and frame_before != frame_after:
                x0, y0 = xs[frame_before], ys[frame_before]
                x1_, y1_ = xs[frame_after], ys[frame_after]
                t = (target_frame - frame_before) / (frame_after - frame_before)
                use_frame = frame_before if abs(target_frame-frame_before)<abs(target_frame-frame_after) else frame_after
            elif frame_before < n_traj:
                t = 0
                use_frame = frame_before
            elif frame_after < n_traj:
                t = 0
                use_frame = frame_after
            else:
                continue

            if frame_before < n_traj and frame_after < n_traj:
                interp_x = xs[frame_before] * (1-t) + xs[frame_after] * t
                interp_y = ys[frame_before] * (1-t) + ys[frame_after] * t
                interp_idx = int(round(target_frame))
            else:
                interp_x = xs[use_frame]
                interp_y = ys[use_frame]
                interp_idx = use_frame
            d = direction[interp_idx]
            line_on_offset = d != 0 and d > 0
            if line_on_offset:
                xuv, yuv = xy_to_uv(interp_x, interp_y)
                yuv_off = yuv + offset_sign * offset_pixels / upscale_factor
                pt = (xuv * upscale_factor, yuv_off * upscale_factor)
            else:
                xuv, yuv = xy_to_uv(interp_x, interp_y)
                pt = (xuv * upscale_factor, yuv * upscale_factor)
            color = color_gradient(s, num_whole_seconds-1 if num_whole_seconds>1 else 1)
            ticks.append({"pt": pt, "color": color, "interp_idx": interp_idx, "second": s, "line_on_offset": line_on_offset})
        return ticks

    ticks = get_tick_points_and_colors()

    for tick in ticks:
        x, y = tick["pt"]
        marker_c = tick["color"]

        cv2.circle(
            high_res_image,
            (int(round(x)), int(round(y))),
            int(marker_radius * upscale_factor // 4 + marker_border_thickness),
            (0, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA
        )
        cv2.circle(
            high_res_image,
            (int(round(x)), int(round(y))),
            int(marker_radius * upscale_factor // 4),
            marker_c,
            thickness=-1,
            lineType=cv2.LINE_AA
        )

        # --- Draw an arrow at this marker ---
        interp_idx = tick["interp_idx"]
        d = direction[interp_idx]
        # Use arrow_length_px for visual arrow size (in original px before upscaling)
        arrow_length = arrow_length_px * upscale_factor
        if arrow_thickness_px is not None:
            arrow_thickness = int(arrow_thickness_px * upscale_factor)
        else:
            arrow_thickness = max(1, marker_radius * upscale_factor // 6)
        angle = 0 if d >= 0 else np.pi  # Right (0 degrees) if d>=0 else left (180 deg)
        dx_arrow = np.cos(angle) * arrow_length * 0.5
        dy_arrow = 0
        tail_x = int(round(x - dx_arrow))
        tip_x = int(round(x + dx_arrow))
        tail_y = tip_y = int(round(y + dy_arrow))
        cv2.arrowedLine(
            high_res_image,
            (tail_x, tail_y),
            (tip_x, tip_y),
            marker_c,
            arrow_thickness,
            tipLength=0.38,
            line_type=cv2.LINE_AA if hasattr(cv2, 'LINE_AA') else 8
        )

        # Blue dotted ring (optional, skip if frame==tick idx)
        if show_blue_dotted_ring and (tick["second"] != frame):
            ball_diameter_world = getattr(jtap_stimulus, 'diameter', None)
            if ball_diameter_world is not None:
                ball_radius_pixels = (ball_diameter_world / 2.0) * pixel_density
            else:
                ball_radius_pixels = marker_radius
            ring_radius = int(round(ball_radius_pixels * upscale_factor))
            ring_color = (0, 0, 255)  # BGR
            num_dots = min(16, int(2 * np.pi * ring_radius / 3))
            dot_radius = 1
            for i_dot in range(num_dots):
                theta = 2 * np.pi * i_dot / num_dots
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

        if not do_not_draw_seconds:
            def find_clear_position(img, x, y, text, search_radius=20):
                h, w, _ = img.shape
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, int(font_thickness * upscale_factor))[0]
                for r in range(1, search_radius):
                    for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r)]:
                        nx, ny = int(x + dx), int(y + dy)
                        x_end, y_end = nx + (text_size[0] + 3), ny - (text_size[1] + 3)
                        if 0 <= nx < w and 0 <= ny < h and 0 <= x_end < w and 0 <= y_end < h:
                            region = img[ny - (text_size[1] + 3):ny, nx:x_end]
                            if not np.any(np.all(region < 27, axis=-1)):
                                return nx, ny
                return x, y
            x_clear, y_clear = find_clear_position(high_res_image, x, y - 15, str(tick["second"]))
            y_adjust = 0
            cv2.putText(
                high_res_image,
                str(tick["second"]),
                (int(x_clear), int(y_clear - y_adjust)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=int(font_thickness * upscale_factor),
                lineType=cv2.LINE_AA
            )

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