import numpy as np
import json
from typing import NamedTuple
import os
import pandas as pd
import pickle

class HumanData(NamedTuple):
    human_keypresses: np.ndarray
    human_output: np.ndarray

class JTAPMiceStimulus(NamedTuple):
    name: str   
    human_data: HumanData
    discrete_obs: np.ndarray
    is_occlusion_trial: bool
    partially_occluded_bool: np.ndarray
    fully_occluded_bool: np.ndarray
    ground_truth_positions: np.ndarray
    num_frames: int
    diameter: float
    fps: int
    skip_t: int
    pixel_density: int


def load_red_green_stimulus(
    stimulus_path, 
    pixel_density=10, 
    skip_t=1, 
    rgb_only=False, 
    skip_human_data_timesteps=True,
):
    # Load simulation data
    with open(os.path.join(stimulus_path, 'simulation_data.json'), 'r') as f:
        simulation_data = json.load(f)
    
    # Create video frames
    rgb_frames, partially_occluded_bool, fully_occluded_bool, discrete_obs = create_video_from_simulation_data( # type: ignore
        simulation_data, pixel_density=pixel_density, skip_t=skip_t
    )

    # Check occlusion
    is_occlusion_trial = partially_occluded_bool.any() or fully_occluded_bool.any()

    # Load human data
    if os.path.exists(os.path.join(stimulus_path, 'human_data.csv')):
        with open(os.path.join(stimulus_path, 'human_data.csv'), 'r') as f:
            human_data_df = pd.read_csv(f)
        human_keypresses, human_output = process_human_data(human_data_df)
        if skip_human_data_timesteps:
            # note that the skip is done on the frames dimension
            # which is different for the keypresses and the output
            human_keypresses = human_keypresses[:, ::skip_t]
            human_output = human_output[::skip_t]
        human_data = HumanData(human_keypresses = human_keypresses, human_output = human_output)
    else:
        human_data = None

    name = stimulus_path.split('/')[-1]

    # Extract ground truth positions
    step_data = simulation_data['step_data']
    timesteps = sorted([int(k) for k in step_data.keys()])[::skip_t]
    ground_truth_positions = np.array([
        [step_data[str(t)]['x'], step_data[str(t)]['y']]
        for t in timesteps
    ])
    diameter = simulation_data['target']['size']
    fps = simulation_data['fps']

    # Return logic
    if rgb_only:
        return rgb_frames
    else:
        # Create JTAPMiceStimulus object
        stimulus = JTAPMiceStimulus(name = name, discrete_obs = discrete_obs, partially_occluded_bool = partially_occluded_bool, fully_occluded_bool = fully_occluded_bool, ground_truth_positions = ground_truth_positions, num_frames = len(rgb_frames), diameter = diameter, fps = fps, skip_t = skip_t, pixel_density = pixel_density, human_data = human_data, is_occlusion_trial = is_occlusion_trial) # type: ignore
        
        return stimulus

def jtap_compute_outputs(keypresses_array):
    green_accuracy = np.mean(keypresses_array == 0, axis = 0)
    red_accuracy = np.mean(keypresses_array == 1, axis = 0)
    uncertain_accuracy = np.mean(keypresses_array == 2, axis = 0)
    return np.stack([green_accuracy, red_accuracy, uncertain_accuracy]).T


def process_human_data(human_data_df):
    """
    Process human data DataFrame and convert to keypresses array.
    
    Args:
        human_data (pd.DataFrame): DataFrame with columns: session_id, frame, green, red, uncertain
        
    Returns:
        np.ndarray: Array of shape (N, T) where N is number of sessions and T is number of frames.
                   Values are 0 (green), 1 (red), 2 (uncertain), or -1 (missing data).
    """
    # Get unique session IDs and max frame number
    session_ids = sorted(human_data_df['session_id'].unique())
    max_frame = human_data_df['frame'].max()
    N = len(session_ids)
    T = max_frame + 1  # +1 because frames start from 0

    # Initialize the N x T array
    human_keypresses = np.full((N, T), -1, dtype=int)  # -1 as placeholder for missing data

    # Create session_id to index mapping for faster lookup
    session_to_idx = {session_id: i for i, session_id in enumerate(session_ids)}
    
    # Convert DataFrame columns to numpy arrays for vectorized operations
    session_ids_array = human_data_df['session_id'].values
    frames_array = human_data_df['frame'].values
    green_array = human_data_df['green'].values
    red_array = human_data_df['red'].values
    uncertain_array = human_data_df['uncertain'].values
    
    # Vectorized assert check: exactly one of the three fields should be 1
    sums = green_array + red_array + uncertain_array
    if not np.all(sums == 1):
        bad_indices = np.where(sums != 1)[0]
        for idx in bad_indices:
            session_id = session_ids_array[idx]
            frame = frames_array[idx]
            assert False, f"Session {session_id}, frame {frame}: exactly one of green/red/uncertain should be 1"
    
    # Map session_ids to indices vectorized
    session_indices = np.array([session_to_idx[sid] for sid in session_ids_array])
    
    # Calculate keypress values vectorized: 0 for green, 1 for red, 2 for uncertain
    keypress_values = green_array * 0 + red_array * 1 + uncertain_array * 2
    
    # Fill the array using advanced indexing
    human_keypresses[session_indices, frames_array] = keypress_values

    human_output = jtap_compute_outputs(human_keypresses)
    
    return human_keypresses, human_output

def rgb_to_discrete_obs(rgb_video_original, skip_t = 1):
    """
    Convert RGB video frames to discrete pixel values for JTAP stimulus processing.
    
    This function maps RGB color values to discrete integer labels based on color thresholds.
    It's designed to work with JTAP simulation videos where specific colors represent
    different scene elements.
    
    Args:
        rgb_video_original (np.ndarray or list): RGB video frames of shape (T, H, W, 3)
                                               where T is number of frames, H is height,
                                               W is width, and 3 is RGB channels.
                                               Values should be in range [0, 255].
        skip_t (int, optional): Frame skip factor. If > 1, only every skip_t-th frame
                              is processed. Defaults to 1 (process all frames).
    
    Returns:
        np.ndarray: Discrete pixel array of shape (T', H, W) where T' = T//skip_t.
                   Values are int8 with the following mapping:
                   - 0: Background (white pixels)
                   - 1: Gray (occluders, RGB ~128)
                   - 2: Blue (target ball, RGB ~[0,0,255])
                   - 3: Black (barriers, RGB ~[0,0,0])
                   - 4: Red (red sensor, RGB ~[255,0,0])
                   - 5: Green (green sensor, RGB ~[0,255,0])
    """
    # Skip frames if needed
    if skip_t > 1:
        rgb_video = rgb_video_original[::skip_t]
    else:
        rgb_video = rgb_video_original
    
    # Initialize output array (starts as background = 0)
    discrete_obs = np.zeros(rgb_video.shape[:3], dtype=np.int8)
    
    # Extract RGB channels for vectorized operations
    r, g, b = rgb_video[..., 0], rgb_video[..., 1], rgb_video[..., 2]
    
    # Apply all masks in order of priority (later assignments override earlier ones)
    
    # Gray: RGB around 128
    discrete_obs[(r >= 118) & (r <= 138) & (g >= 118) & (g <= 138) & (b >= 118) & (b <= 138)] = 1
    
    # Black: RGB < 50
    discrete_obs[(r < 50) & (g < 50) & (b < 50)] = 3
    
    # Blue: low R,G, high B
    discrete_obs[(r < 100) & (g < 100) & (b > 220)] = 2
    
    # Red: high R, low G,B  
    discrete_obs[(r > 220) & (g < 100) & (b < 100)] = 4
    
    # Green: low R,B, high G
    discrete_obs[(r < 100) & (g > 220) & (b < 100)] = 5
    
    return discrete_obs


def discrete_obs_to_rgb(discrete_obs):
    """
    Convert discrete pixel values back to RGB video frames.
    
    This function performs the inverse operation of rgb_to_discrete_obs,
    converting discrete integer labels back to RGB color values for visualization
    or further processing.
    
    Args:
        discrete_obs (np.ndarray): Discrete pixel array of shape (T, H, W) with
                                 integer values representing different scene elements.
                                 Expected dtype is int8 or compatible integer type.
                                 Values should be in range [0, 5].
    
    Returns:
        np.ndarray: RGB video array of shape (T, H, W, 3) with uint8 values
                   in range [0, 255]. Color mapping:
                   - 0: White background [255, 255, 255]
                   - 1: Gray occluders [128, 128, 128]
                   - 2: Blue target [0, 0, 255]
                   - 3: Black barriers [0, 0, 0]
                   - 4: Red sensor [255, 0, 0]
                   - 5: Green sensor [0, 255, 0]
    
    Note:
        This function assumes the input follows the discrete pixel encoding
        used by JTAP stimulus processing pipeline.
    """
    # Convert to numpy array if not already
    discrete_obs = np.array(discrete_obs)
    
    # Initialize RGB output array
    rgb_video = np.zeros((*discrete_obs.shape, 3), dtype=np.uint8)
    
    # Create color mapping - background (0) defaults to black, will be set to white
    rgb_video[discrete_obs == 0] = [255, 255, 255]  # White background
    rgb_video[discrete_obs == 1] = [128, 128, 128]  # Gray
    rgb_video[discrete_obs == 2] = [0, 0, 255]      # Blue (target)
    rgb_video[discrete_obs == 3] = [0, 0, 0]        # Black (barriers)
    rgb_video[discrete_obs == 4] = [255, 0, 0]      # Red (red sensor)
    rgb_video[discrete_obs == 5] = [0, 255, 0]      # Green (green sensor)
    
    return rgb_video


def create_video_from_simulation_data(simulation_data, pixel_density=20, skip_t = 1):
    """
    Convert simulation data to RGB video frames.
    
    Creates RGB video frames from JTAP simulation data. The scene coordinate system uses:
    - Origin (0,0) at bottom-left of scene
    - X-axis increases rightward
    - Y-axis increases upward
    
    Video frames use standard image indexing:
    - frame[y, x, channel] where y=0 is top row, x=0 is left column
    - Channels: 0=Red, 1=Green, 2=Blue
    
    Color mapping:
    - White (255,255,255): Background
    - Black (0,0,0): Barriers  
    - Red (255,0,0): Red sensor
    - Green (0,255,0): Green sensor
    - Blue (0,0,255): Target object
    - Gray (128,128,128): Occluders
    """
    # Extract scene dimensions

    # important to round the values to 2 decimal places
    # otherwise the entities may be off by 1 pixel
    rnd = lambda x : round(x,2)

    scene_width, scene_height = simulation_data['scene_dims']
    
    # Get step data and sort by timestep
    step_data = simulation_data['step_data']
    timesteps = sorted([int(k) for k in step_data.keys()])[::skip_t]
    
    # Precompute pixel dimensions
    frame_height = scene_height * pixel_density
    frame_width = scene_width * pixel_density
    
    # Create base frame with static elements (white background, barriers, sensors)
    base_frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
    discrete_obs_base_frame = np.zeros((frame_height, frame_width), dtype=np.int8)
    
    # Draw barriers (black rectangles) on base frame
    for barrier in simulation_data['barriers']:
        x, y, width, height = rnd(barrier['x']), rnd(barrier['y']), rnd(barrier['width']), rnd(barrier['height'])
        # Convert to pixel coordinates
        # Flip Y coordinate: y_flipped = scene_height - y - height
        x_px = max(int(x * pixel_density), 0)
        y_px = max(int(rnd(scene_height - y - height) * pixel_density) - 1, 0)
        w_px = int(width * pixel_density)
        h_px = int(height * pixel_density)
        
        # Draw barrier as black rectangle
        base_frame[y_px:y_px+h_px, x_px:x_px+w_px] = 0  # Black
        discrete_obs_base_frame[y_px:y_px+h_px, x_px:x_px+w_px] = 3
    
    # Draw red sensor on base frame
    red_sensor = simulation_data['red_sensor']
    x, y, width, height = rnd(red_sensor['x']), rnd(red_sensor['y']), rnd(red_sensor['width']), rnd(red_sensor['height'])
    # Flip Y coordinate: y_flipped = scene_height - y - height
    x_px = max(int(x * pixel_density), 0)
    y_px = max(int(rnd(scene_height - y - height) * pixel_density) - 1, 0)
    w_px = int(width * pixel_density)
    h_px = int(height * pixel_density)
    base_frame[y_px:y_px+h_px, x_px:x_px+w_px, 0] = 255  # Red
    base_frame[y_px:y_px+h_px, x_px:x_px+w_px, 1:] = 0
    discrete_obs_base_frame[y_px:y_px+h_px, x_px:x_px+w_px] = 4

    # Draw green sensor on base frame
    green_sensor = simulation_data['green_sensor']
    x, y, width, height = rnd(green_sensor['x']), rnd(green_sensor['y']), rnd(green_sensor['width']), rnd(green_sensor['height'])
    # Flip Y coordinate: y_flipped = scene_height - y - height
    x_px = max(int(x * pixel_density), 0)
    y_px = max(int(rnd(scene_height - y - height) * pixel_density) - 1, 0)
    w_px = int(width * pixel_density)
    h_px = int(height * pixel_density)
    base_frame[y_px:y_px+h_px, x_px:x_px+w_px, 1] = 255  # Green
    base_frame[y_px:y_px+h_px, x_px:x_px+w_px, [0, 2]] = 0
    discrete_obs_base_frame[y_px:y_px+h_px, x_px:x_px+w_px] = 5
    
    # Precompute all static elements for faster copying
    num_frames = len(timesteps)
    rgb_video = np.empty((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    discrete_obs_video = np.empty((num_frames, frame_height, frame_width), dtype=np.int8)
    partially_occluded_bool = np.empty(num_frames, dtype=bool)
    fully_occluded_bool = np.empty(num_frames, dtype=bool)

    # Precompute occluder rectangles in scene coordinates and pixel coordinates
    occluders = simulation_data['occluders']
    occluder_rects = []
    occluder_pixel_rects = []
    for occ in occluders:
        # Scene coordinates
        x0 = rnd(occ['x'])
        y0 = rnd(occ['y'])
        x1 = x0 + rnd(occ['width'])
        y1 = y0 + rnd(occ['height'])
        occluder_rects.append((x0, y0, x1, y1))
        
        # Pixel coordinates for faster rendering
        x_px = max(int(x0 * pixel_density), 0)
        y_px = max(int(rnd(scene_height - y0 - occ['height']) * pixel_density) - 1, 0)
        w_px = int(rnd(occ['width']) * pixel_density)
        h_px = int(rnd(occ['height']) * pixel_density)
        occluder_pixel_rects.append((y_px, y_px+h_px, x_px, x_px+w_px))

    target_size = rnd(simulation_data['target']['size'])
    target_radius = target_size / 2
    target_radius_px = int(target_radius * pixel_density)
    target_radius_px_sq = target_radius_px ** 2

    def is_circle_intersecting_box(box_x1, box_y1, box_x2, box_y2, circle_x, circle_y, radius):
        # Find the closest point on the rectangle to the circle center
        closest_x = min(max(circle_x, box_x1), box_x2)
        closest_y = min(max(circle_y, box_y1), box_y2)
        dist_sq = (closest_x - circle_x) ** 2 + (closest_y - circle_y) ** 2
        # STRICTLY LESS THAN is when the circle is finally intersecting the box
        return dist_sq < radius ** 2

    for i, timestep in enumerate(timesteps):
        # Start with base frame
        rgb_video[i] = base_frame
        discrete_obs_video[i] = discrete_obs_base_frame
        
        # Draw target (blue circle)
        step = step_data[str(timestep)]
        target_x, target_y = step['x'], step['y']
        # The x, y position refers to bottom left of bounding box, so add radius to get center
        target_center_x = target_x + target_radius
        target_center_y = target_y + target_radius

        target_center_x_px = int(target_center_x * pixel_density)
        target_center_y_px = int((scene_height - target_center_y) * pixel_density)  # Flip Y

        # Create circle mask more efficiently
        y_min = max(0, target_center_y_px - target_radius_px)
        y_max = min(frame_height, target_center_y_px + target_radius_px + 1)
        x_min = max(0, target_center_x_px - target_radius_px)
        x_max = min(frame_width, target_center_x_px + target_radius_px + 1)
        
        # Only compute mask for relevant region
        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x_coords - target_center_x_px + 0.5)**2 + (y_coords - target_center_y_px + 0.5)**2 < target_radius_px_sq
        
        # Apply target color to relevant region
        rgb_video[i, y_min:y_max, x_min:x_max][mask, 2] = 255  # Blue
        rgb_video[i, y_min:y_max, x_min:x_max][mask, :2] = 0   # Clear red/green
        discrete_obs_video[i, y_min:y_max, x_min:x_max][mask] = 2
        
        # Draw occluders (gray rectangles) - must be rendered after target to occlude it
        for y_start, y_end, x_start, x_end in occluder_pixel_rects:
            # Draw occluder as gray rectangle
            rgb_video[i, y_start:y_end, x_start:x_end] = 128  # Gray
            discrete_obs_video[i, y_start:y_end, x_start:x_end] = 1
        
        # Check if there are any blue pixels (target visible) - more efficient check
        is_fully_occluded = not np.any(discrete_obs_video[i] == 2)

        # --- Compute occlusion booleans using ground truth geometry (target is a circle) ---
        # Partially occluded: the circle intersects any occluder, but is not fully occluded
        is_partially_occluded = False
        if not is_fully_occluded:
            for (ox0, oy0, ox1, oy1) in occluder_rects:
                if is_circle_intersecting_box(ox0, oy0, ox1, oy1, target_center_x, target_center_y, target_radius):
                    is_partially_occluded = True
                    break

        fully_occluded_bool[i] = is_fully_occluded
        partially_occluded_bool[i] = is_partially_occluded

        assert not (is_fully_occluded and is_partially_occluded), "Target is both fully and partially occluded"

    return rgb_video, partially_occluded_bool, fully_occluded_bool, discrete_obs_video

#########################################
# Left Right Stimulus
#########################################

def discrete_lr_obs_to_rgb(discrete_obs):
    """
    Convert discrete pixel values back to RGB video frames for Left Right Stimulus

    Args:
        discrete_obs (np.ndarray): Discrete pixel array of shape (T, H, W) with
                                 integer values representing different scene elements.
                                 Expected dtype is int8 or compatible integer type.
                                 Values should be in range [0, 2].

    Returns:
        np.ndarray: RGB video array of shape (T, H, W, 3) with uint8 values
                   in range [0, 255]. Color mapping:
                   - 0: White background [255, 255, 255]
                   - 1: Gray occluders [128, 128, 128]
                   - 2: Blue (target) [0, 0, 255]

    Note:
        This function assumes the input follows the discrete pixel encoding
        used by JTAP stimulus processing pipeline.
    """
    # Convert to numpy array if not already
    discrete_obs = np.array(discrete_obs)
    
    # Initialize RGB output array
    rgb_video = np.zeros((*discrete_obs.shape, 3), dtype=np.uint8)
    
    # Create color mapping - background (0) defaults to black, will be set to white
    rgb_video[discrete_obs == 0] = [255, 255, 255]  # White background
    rgb_video[discrete_obs == 1] = [128, 128, 128]  # Gray
    rgb_video[discrete_obs == 2] = [0, 0, 255]      # Blue (target)

    return rgb_video


#########################################
# Original JTAP Results loading
#########################################

def load_original_jtap_results():
    """
    Load original JTAP results from a pickle file

    Args:
        None

    Returns:
        original_jtap_results (dict): Dictionary containing the original JTAP results
    """

    from jtap_mice.utils import get_assets_dir
    
    original_jtap_results_path = os.path.join(get_assets_dir(), 'original_jtap_data', 'original_jtap_results.pkl')
    with open(original_jtap_results_path, 'rb') as f:
        original_jtap_results = pickle.load(f)
    return original_jtap_results