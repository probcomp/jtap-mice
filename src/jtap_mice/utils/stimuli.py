import numpy as np
import json
from typing import NamedTuple, Optional
import os

class MouseData(NamedTuple):
    pass

class JTAPMiceStimulus(NamedTuple):
    name: str   
    mouse_data: Optional[MouseData]
    discrete_obs: np.ndarray
    is_occlusion_trial: bool
    is_switching_trial: bool
    scene_length: float
    partially_occluded_bool: np.ndarray
    fully_occluded_bool: np.ndarray
    ground_truth_positions: np.ndarray
    num_frames: int
    diameter: float
    fps: int
    skip_t: int
    pixel_density: int

def load_left_right_stimulus(
    stimulus_path, 
    pixel_density=10, 
    skip_t=1, 
    rgb_only=False, 
    trial_number=None,
):
    """
    Loads a left-right (or mice) stimulus. For new "mice" multi-trial JSONs, supply trial_number.
    """
    # Only support the new-style JSON with trial structure
    if not (os.path.isfile(stimulus_path) and stimulus_path.endswith('.json')):
        raise ValueError("Only new-style .json stimulus files are supported, found: {!r}".format(stimulus_path))
    
    with open(stimulus_path, 'r') as f:
        all_trials_data = json.load(f)

    # trial_number must be specified
    assert trial_number is not None, "trial_number must be specified for new-style mice stimuli JSON"
    trial_data = all_trials_data["trials"][trial_number]
    name = f"{os.path.splitext(os.path.basename(stimulus_path))[0]}_trial{trial_number}"

    # Use info from config - all required keys must exist
    config = all_trials_data["config"]
    scene_length = float(config["scene_length"])
    fps = int(config["fps"])
    diameter = float(config["diameter"])
    scene_width = scene_length
    scene_height = config["scene_height"]
    
    # Get ball positions (centers)
    positions = np.asarray(trial_data["positions"], dtype=np.float32)  # (T,) or (T,1), one X per frame
    if positions.ndim == 2:
        positions = positions.squeeze(-1)
    # Y is always centered vertically (for this mice setup): put ball in the center in Y
    ball_center_y = float(config["ball_y_center"]) if "ball_y_center" in config else float(scene_height) / 2.0
    ground_truth_positions = np.stack([positions, np.full_like(positions, ball_center_y)], axis=1)

    is_switching_trial = bool(trial_data.get("switching", False) or trial_data.get("is_switching", False))
    # No occlusion for now, but could be in the config in future
    occluders = []
    partially_occluded_bool = np.zeros(len(positions), dtype=bool)
    fully_occluded_bool = np.zeros(len(positions), dtype=bool)
    is_occlusion_trial = False

    # Create the video and discrete obs
    rgb_frames, discrete_obs = create_mice_video_from_positions(
        positions=positions,
        scene_width=scene_width,
        scene_height=scene_height,
        diameter=diameter,
        pixel_density=pixel_density,
        skip_t=skip_t,
        ball_center_y=ball_center_y,
    )
    num_frames = len(rgb_frames)

    # By default, no mouse data for these stimuli
    mouse_data = None
    # Final output
    if rgb_only:
        return rgb_frames
    else:
        return JTAPMiceStimulus(
            name = name,
            mouse_data = mouse_data,
            discrete_obs = discrete_obs,
            is_occlusion_trial = is_occlusion_trial,
            is_switching_trial = is_switching_trial,
            scene_length = scene_length,
            partially_occluded_bool = partially_occluded_bool,
            fully_occluded_bool = fully_occluded_bool,
            ground_truth_positions = ground_truth_positions,
            num_frames = num_frames,
            diameter = diameter,
            fps = fps,
            skip_t = skip_t,
            pixel_density = pixel_density
        )

def create_mice_video_from_positions(
    positions,
    scene_width=100.0,
    scene_height=20.0,
    diameter=11.0,
    pixel_density=10,
    skip_t=1,
    ball_center_y=None,
):
    """
    Create video and discrete_obs for the 'mice' left-right type stimulus.
    - positions are X locations of the BALL CENTER.
    - Y will be constant and centered unless specified.
    """
    # Quantize dimensions
    frame_width = int(np.round(scene_width * pixel_density))
    frame_height = int(np.round(scene_height * pixel_density))
    T = len(positions)
    rgb_frames = np.full((T, frame_height, frame_width, 3), 255, dtype=np.uint8)  # white background
    discrete_obs = np.zeros((T, frame_height, frame_width), dtype=np.int8)  # background=0

    ball_radius = diameter / 2.
    ball_radius_px = int(np.round(ball_radius * pixel_density))
    ball_center_y = float(scene_height / 2.0) if ball_center_y is None else float(ball_center_y)
    center_y_px = int(np.round((scene_height - ball_center_y) * pixel_density))

    for i, xc in enumerate(positions):
        # Ball center in pixel
        center_x_px = int(np.round(xc * pixel_density))
        # Draw filled circle (blue for rgb, 2 for discrete)
        y_min = max(0, center_y_px - ball_radius_px)
        y_max = min(frame_height, center_y_px + ball_radius_px + 1)
        x_min = max(0, center_x_px - ball_radius_px)
        x_max = min(frame_width, center_x_px + ball_radius_px + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle_mask = (xx - center_x_px + 0.5) ** 2 + (yy - center_y_px + 0.5) ** 2 < (ball_radius_px) ** 2
        # RGB blue
        rgb_frames[i, y_min:y_max, x_min:x_max][circle_mask, 2] = 255
        rgb_frames[i, y_min:y_max, x_min:x_max][circle_mask, :2] = 0
        # Discrete mask: code 2 for ball
        discrete_obs[i, y_min:y_max, x_min:x_max][circle_mask] = 2  # No occluders yet

    return rgb_frames, discrete_obs

def rgb_to_discrete_obs(rgb_video_original, skip_t=1):
    # As before, unchanged
    if skip_t > 1:
        rgb_video = rgb_video_original[::skip_t]
    else:
        rgb_video = rgb_video_original

    discrete_obs = np.zeros(rgb_video.shape[:3], dtype=np.int8)
    r, g, b = rgb_video[..., 0], rgb_video[..., 1], rgb_video[..., 2]
    discrete_obs[(r >= 118) & (r <= 138) & (g >= 118) & (g <= 138) & (b >= 118) & (b <= 138)] = 1
    discrete_obs[(r < 100) & (g < 100) & (b > 220)] = 2
    return discrete_obs

def discrete_obs_to_rgb(discrete_obs):
    discrete_obs = np.array(discrete_obs)
    rgb_video = np.zeros((*discrete_obs.shape, 3), dtype=np.uint8)
    rgb_video[discrete_obs == 0] = [255, 255, 255]  # White background
    rgb_video[discrete_obs == 1] = [128, 128, 128]  # Gray (would be occluder)
    rgb_video[discrete_obs == 2] = [0, 0, 255]      # Blue (target)
    return rgb_video