import numpy as np
import json
from typing import NamedTuple, Optional, List
import os

class MouseData(NamedTuple):
    pass

class JTAPMiceStimulus(NamedTuple):
    name: str   
    trial_number: int
    mouse_data: Optional[MouseData]
    discrete_obs: np.ndarray
    is_occlusion_trial: bool
    is_switching_trial: bool
    occlusion_regions: List
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
    inject_occlusion=None,
):
    """
    Loads a left-right (or mice) stimulus. For new "mice" multi-trial JSONs, supply trial_number.
    Applies temporal downsampling (frame skipping) via skip_t to all frame-wise arrays.
    fps ALWAYS remains the original fps (NOT downsampled fps in structure).
    
    Args:
        inject_occlusion: Optional 2D array/list of shape (N_OCC, 2) where each row is [x_left, length]
                         representing occluders. x_left is the left edge position, length is the width.
    """
    # Only support the new-style JSON with trial structure
    if not (os.path.isfile(stimulus_path) and stimulus_path.endswith('.json')):
        raise ValueError(f"Only new-style .json stimulus files are supported, found: {stimulus_path!r}")

    with open(stimulus_path, 'r') as f:
        all_trials_data = json.load(f)

    # trial_number must be specified
    assert trial_number is not None, "trial_number must be specified for new-style mice stimuli JSON"

    # ---- Extraction from JSON, following example lr_v1.json ----
    # Top level: {"config": {...}, "trial_data": { "1": [...], ... } }
    config = all_trials_data["config"]

    trial_key = str(trial_number)
    if "trial_data" not in all_trials_data:
        raise ValueError(f"JSON is missing 'trial_data' key. Found keys: {list(all_trials_data.keys())}")
    trial_data_dict = all_trials_data["trial_data"]
    if trial_key not in trial_data_dict:
        raise ValueError(f"Trial {trial_key} not found in 'trial_data' in stimulus file.")

    positions = np.asarray(trial_data_dict[trial_key], dtype=np.float32) 
    # positions shape (T,), represents X coordinates per frame

    # Name: same format as before, use file name + trial number
    name = f"{os.path.splitext(os.path.basename(stimulus_path))[0]}_trial_{trial_key}"

    # Use info from config - all required keys should exist per lr_v1.json
    # scene_length is called "LEFT_RIGHT_LENGTH" in lr_v1.json, see example
    # diameter is called "SPEED" (probably not), so use a fallback or a default if not present
    scene_length = float(config["LEFT_RIGHT_LENGTH"])
    fps = int(config["FRAMES_PER_SECOND"])
    diameter = 1.0
    scene_width = scene_length
    # scene_height MAY not exist, so set to a default value
    scene_height = diameter
    ball_bottom_y = 0.0
    ball_center_y = diameter / 2.0

    # Validate inject_occlusion if provided
    if inject_occlusion is not None:
        inject_occlusion = np.asarray(inject_occlusion, dtype=np.float32)
        if inject_occlusion.ndim != 2 or inject_occlusion.shape[1] != 2:
            raise ValueError(f"inject_occlusion must be a 2D array with shape (N_OCC, 2), got shape {inject_occlusion.shape}")
        
        # Check bounds for each occluder
        for i, (x_left, length) in enumerate(inject_occlusion):
            if x_left < 0 or x_left > scene_width:
                raise ValueError(f"Occluder {i}: left edge position {x_left} is outside scene bounds [0, {scene_width}]")
            if x_left + length < 0 or x_left + length > scene_width:
                raise ValueError(f"Occluder {i}: right edge position {x_left + length} is outside scene bounds [0, {scene_width}]")

    # construct ground_truth_positions (T, 2): [x, y=ball_bottom_y]
    ground_truth_positions_full = np.stack([positions - (diameter/2.0), np.full_like(positions, ball_bottom_y)], axis=1)

    # is_switching is not in lr_v1.json's trial_data, so set to False unless inference needed (not available in current JSON)
    # Check switching BEFORE skip, based on original positions
    if len(positions) > 1:
        velocities = np.diff(positions)
        sign_changes = np.where(np.diff(np.sign(velocities)) != 0)[0]
        is_switching_trial = len(sign_changes) > 0
    else:
        is_switching_trial = False
    
    # Set occlusion trial flag based on whether occlusion is injected
    is_occlusion_trial = inject_occlusion is not None

    # FULL per-frame arrays (frame-count == positions.shape[0])
    partially_occluded_bool_full = np.zeros(len(positions), dtype=bool)
    fully_occluded_bool_full = np.zeros(len(positions), dtype=bool)

    # If occlusion is injected, compute occlusion masks
    if inject_occlusion is not None:
        ball_radius = diameter / 2.0
        for frame_idx, ball_center_x in enumerate(positions):
            ball_left = ball_center_x - ball_radius
            ball_right = ball_center_x + ball_radius
            
            for x_left, length in inject_occlusion:
                occ_right = x_left + length
                
                # Check for overlap between ball and occluder
                overlap_left = max(ball_left, x_left)
                overlap_right = min(ball_right, occ_right)
                
                if overlap_left < overlap_right:  # There is overlap
                    overlap_fraction = (overlap_right - overlap_left) / diameter
                    if overlap_fraction >= 1.0:
                        fully_occluded_bool_full[frame_idx] = True
                    else:
                        partially_occluded_bool_full[frame_idx] = True

    # Apply ::skip_t everywhere
    positions_sub = positions[::skip_t]
    ground_truth_positions = ground_truth_positions_full[::skip_t]
    partially_occluded_bool = partially_occluded_bool_full[::skip_t]
    fully_occluded_bool = fully_occluded_bool_full[::skip_t]

    rgb_frames, discrete_obs = create_mice_video_from_positions(
        positions=positions_sub,
        scene_width=scene_width,
        scene_height=scene_height,
        diameter=diameter,
        pixel_density=pixel_density,
        skip_t=1,  # already applied skip_t to positions, so pass-through 1 here
        ball_center_y=ball_center_y,
        inject_occlusion=inject_occlusion,
    )
    num_frames = len(rgb_frames)

    mouse_data = None

    if rgb_only:
        return rgb_frames
    else:
        return JTAPMiceStimulus(
            name=name,
            trial_number=int(trial_key),
            mouse_data=mouse_data,
            discrete_obs=discrete_obs,
            occlusion_regions=inject_occlusion,
            is_occlusion_trial=is_occlusion_trial,
            is_switching_trial=is_switching_trial,
            scene_length=scene_length,
            partially_occluded_bool=partially_occluded_bool,
            fully_occluded_bool=fully_occluded_bool,
            ground_truth_positions=ground_truth_positions,
            num_frames=num_frames,
            diameter=diameter,
            fps=fps,
            skip_t=skip_t,
            pixel_density=pixel_density
        )

def create_mice_video_from_positions(
    positions,
    scene_width=100.0,
    scene_height=20.0,
    diameter=11.0,
    pixel_density=10,
    skip_t=1,
    ball_center_y=None,
    inject_occlusion=None,
):
    """
    Create video and discrete_obs for the 'mice' left-right type stimulus.
    - positions are X locations of the BALL CENTER.
    - Y will be constant and centered unless specified.
    skip_t in this function should always be 1; frame skipping should be done before passing to this function.
    
    Args:
        inject_occlusion: Optional 2D array of shape (N_OCC, 2) where each row is [x_left, length]
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
        discrete_obs[i, y_min:y_max, x_min:x_max][circle_mask] = 2  # Ball overwrites occluders

    if inject_occlusion is not None:
        for x_left, length in inject_occlusion:
            occ_left_px = int(np.round(x_left * pixel_density))
            occ_right_px = int(np.round((x_left + length) * pixel_density))
            occ_left_px = max(0, occ_left_px)
            occ_right_px = min(frame_width, occ_right_px)
            
            # Draw occluder for all frames (gray color, discrete value 1)
            for i in range(T):
                rgb_frames[i, :, occ_left_px:occ_right_px] = [128, 128, 128]  # gray
                discrete_obs[i, :, occ_left_px:occ_right_px] = 1  # occluder code

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