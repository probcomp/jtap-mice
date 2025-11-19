#!/usr/bin/env python3
"""
Simulation Runner Script

This script converts init_state_entities.json to simulation_data.json by running
a physics simulation using pymunk.

Usage:
    # Single trial mode
    uv run python scripts/run_simulation.py --stimulus_path assets/stimuli/cogsci_2025_trials/E50
    
    # Batch processing mode
    uv run python scripts/run_simulation.py --stimulus_folder assets/stimuli/cogsci_2025_trials --batch_mode
    
    # Custom JSON filename
    uv run python scripts/run_simulation.py --stimulus_path assets/stimuli/cogsci_2025_trials/E50 --json_filename my_config.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pymunk
from tqdm import tqdm

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

# Set up rich console, fallback to print if unavailable
if Console is not None:
    console = Console()
    def p(*args, **kwargs):
        console.print(*args, **kwargs)
else:
    console = None
    def p(*args, **kwargs):
        print(*args, **kwargs)

# Try to import distractor functions if they exist
try:
    from jtap.simulation.distractors import simulate_key_distractor, generate_random_distractors
    DISTRACTOR_FUNCTIONS_AVAILABLE = True
except ImportError:
    DISTRACTOR_FUNCTIONS_AVAILABLE = False
    def simulate_key_distractor(*args, **kwargs):
        raise NotImplementedError("simulate_key_distractor not available")
    def generate_random_distractors(*args, **kwargs):
        raise NotImplementedError("generate_random_distractors not available")

def pretty_panel(title, value=None, subtitle=None, style="bold green", highlight=True):
    if console is not None:
        panel_text = Text(str(value) if value is not None else "")
        panel = Panel(panel_text, title=str(title), subtitle=str(subtitle) if subtitle else None, style="green", highlight=highlight)
        console.print(panel)
        return panel
    else:
        return f"[{title}] {value}" if value is not None else f"[{title}]"

def pretty_rule(title, style="bold magenta"):
    if console is not None:
        console.rule(Text(str(title)))
    else:
        print("=" * 10 + f" {title} " + "=" * 10)

def pretty_warning(msg):
    if console is not None:
        console.print(f"[bold yellow]⚠️  {msg}[/bold yellow]")
    else:
        print("WARNING:", msg)

def pretty_error(msg):
    if console is not None:
        console.print(f"[bold red]❌ {msg}[/bold red]")
    else:
        print("ERROR:", msg)

def pretty_info(msg, style="bold blue"):
    if console is not None:
        if isinstance(msg, str):
            msg_stripped = msg.replace("[italic]", "").replace("[/italic]", "").replace("[blue]", "").replace("[/blue]", "")
            console.print(Text(msg_stripped))
        else:
            console.print(msg)
    else:
        print(msg)

def pretty_success(msg):
    if console is not None:
        if isinstance(msg, Panel):
            console.print(msg)
        else:
            msg_stripped = str(msg).replace("[italic]", "").replace("[/italic]", "")
            console.print(f"[bold green]✅ {msg_stripped}[/bold green]")
    else:
        print(msg)

def run_simulation_with_visualization(entities, simulationParams, distractorParams=None):
    """
    Run physics simulation using pymunk and return simulation data.
    
    Args:
        entities: List of entity dictionaries
        simulationParams: Dictionary with keys: videoLength, ballSpeed, fps, physicsStepsPerFrame, res_multiplier, timestep, worldWidth, worldHeight
        distractorParams: Optional dictionary with distractor parameters
    
    Returns:
        Dictionary containing simulation data
    """
    videoLength, ballSpeed, fps, physicsStepsPerFrame, res_multiplier, timestep, worldWidth, worldHeight = simulationParams
    # Calculate derived values
    numFrames = int(videoLength * fps)
    FRAME_INTERVAL = physicsStepsPerFrame
    TIMESTEP = timestep
    FPS = fps
    
    sim_data = {
        'barriers': [], 
        'occluders': [], 
        'step_data': {}, 
        'rg_hit_timestep': -1, 
        'rg_outcome': None,
        'key_distractors': [],
        'random_distractors': []
    }
    
    # Pymunk simulation constants
    GRAVITY = (0, 0)  # Example gravity vector
    interval = 0.1/res_multiplier  # Pixels in the 2D visualization space
    friction = 0.0
    elasticity = 1.0
    
    sim_data['scene_dims'] = (worldWidth, worldHeight)
    sim_data['interval'] = interval
    sim_data['friction'] = friction
    sim_data['elasticity'] = elasticity
    sim_data['timestep'] = TIMESTEP
    sim_data['timesteps_per_frame'] = FRAME_INTERVAL
    sim_data['num_frames'] = numFrames
    sim_data['fps'] = FPS
    
    SPACE_SIZE_width = worldWidth * int(1/interval)
    SPACE_SIZE_height = worldHeight * int(1/interval)
    pix_x = np.arange(0, worldWidth, interval)
    pix_y = np.arange(0, worldHeight, interval)
    y_vals, x_vals = np.meshgrid(pix_x, pix_y, indexing='ij')
    y_vals = np.flip(y_vals)
    
    # Initialize Pymunk space
    space = pymunk.Space()
    static_body = space.static_body
    walls = [
        pymunk.Segment(static_body, (0, 0), (worldWidth, 0), 0.01),  # Bottom
        pymunk.Segment(static_body, (0, 0), (0, worldHeight), 0.01),  # Left
        pymunk.Segment(static_body, (worldWidth, 0), (worldWidth, worldHeight), 0.01),  # Right
        pymunk.Segment(static_body, (0, worldHeight), (worldWidth, worldHeight), 0.01),  # Top
    ]
    
    for wall in walls:
        wall.elasticity = elasticity
        wall.friction = friction
        space.add(wall)
    
    space.gravity = GRAVITY
    
    # Map for storing Pymunk bodies and shapes
    body_map = {}
    
    # Add entities as Pymunk bodies and shapes
    for entity in entities:
        valid_physics_entity = False
        x, y, width, height = entity["x"], entity["y"], entity["width"], entity["height"]
        
        if entity["type"] == "target":
            valid_physics_entity = True
            direction = entity['direction']
            # The ball speed is controlled by physics parameters (timestep and physicsStepsPerFrame)
            # not by the initial velocity magnitude
            vx, vy = np.cos(direction), np.sin(direction)
            # Circle for the target
            radius = width / 2
            mass = 1.0  # Assign a reasonable mass
            moment = pymunk.moment_for_circle(mass, 0, radius)  # Calculate moment of inertia
            body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
            body.position = (x + radius, y + radius)
            body.velocity = (vx, vy)
            shape = pymunk.Circle(body, radius)
            sim_data['target'] = {'size' : width, 'shape' : 1} # 0 for square and 1 for circle
            
        elif entity["type"] == "barrier":
            valid_physics_entity = True
            # Rectangles for other entities
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x + width / 2, y + height / 2)
            shape = pymunk.Poly.create_box(body, (width, height))
            sim_data['barriers'].append({'x' : x,
                                        'y' : y,
                                        'width' : width,
                                        'height' : height})
            
        elif entity["type"] == 'occluder':
            sim_data['occluders'].append({'x' : x,
                                        'y' : y,
                                        'width' : width,
                                        'height' : height})
            
        elif entity["type"] == 'green_sensor':
            sim_data['green_sensor'] = {'x' : x,
                                        'y' : y,
                                        'width' : width,
                                        'height' : height}
            
        elif entity["type"] == 'red_sensor':
            sim_data['red_sensor'] = {'x' : x,
                                        'y' : y,
                                        'width' : width,
                                        'height' : height}
        
        if valid_physics_entity:
            # Add shape to space
            shape.elasticity = elasticity  # Example elasticity
            shape.friction = friction  # Example friction
            space.add(body, shape)
            body_map[entity["id"]] = (body, shape)
    
    sim_data['num_barriers'] = len(sim_data['barriers'])
    sim_data['num_occs'] = len(sim_data['occluders'])
    
    has_hit_red_green = False
    tx, ty = None, None  # Initialize to avoid NameError
    
    # Simulate for the given number of frames
    for frame in range(numFrames):
        if frame != 0:
            for _ in range(FRAME_INTERVAL):
                space.step(TIMESTEP)
        
        # Draw the entities in the frame
        for entity in entities:
            if entity['id'] in list(body_map.keys()):
                body, shape = body_map[entity["id"]]
                
                if isinstance(shape, pymunk.Circle):
                    r = shape.radius
                    tx, ty  = body.position.x - r, body.position.y - r
                    vx, vy  = body.velocity.x, body.velocity.y
                    speed = np.sqrt(vx**2 + vy**2)
                    direction = np.arctan2(vy,vx)
                    
                    target_mask = np.square(x_vals + interval/2 - (tx + r)) + np.square(y_vals + interval/2 - (ty + r)) <= np.square(r)
                    
                    # save to metadata
                    sim_data['step_data'][frame] = { # overspecified
                        'x' : tx,
                        'y' : ty,
                        'speed' : speed,
                        'dir' : direction,
                        'vx' : vx,
                        'vy' : vy
                    }
        
        # NOTE: NEED TO PROCESS THIS IN RED AND IN GREEN!!!!!
        if 'red_sensor' in sim_data or 'green_sensor' in sim_data:
            if tx is None or ty is None:
                # Skip sensor detection if target position not available
                in_red = False
                in_green = False
            else:
                if 'red_sensor' in sim_data:
                    radius = sim_data['target']['size'] / 2
                    center_x = tx + radius
                    center_y = ty + radius
                    red_sensor = sim_data['red_sensor']
                    # Check if circle overlaps with rectangle
                    closest_x = max(red_sensor['x'], min(center_x, red_sensor['x'] + red_sensor['width']))
                    closest_y = max(red_sensor['y'], min(center_y, red_sensor['y'] + red_sensor['height']))
                    distance_sq = (center_x - closest_x)**2 + (center_y - closest_y)**2
                    in_red = distance_sq <= radius**2
                else:
                    in_red = False
                    
                if 'green_sensor' in sim_data:
                    radius = sim_data['target']['size'] / 2
                    center_x = tx + radius
                    center_y = ty + radius
                    green_sensor = sim_data['green_sensor']
                    # Check if circle overlaps with rectangle
                    closest_x = max(green_sensor['x'], min(center_x, green_sensor['x'] + green_sensor['width']))
                    closest_y = max(green_sensor['y'], min(center_y, green_sensor['y'] + green_sensor['height']))
                    distance_sq = (center_x - closest_x)**2 + (center_y - closest_y)**2
                    in_green = distance_sq <= radius**2
                else:
                    in_green = False
        else:
            in_red = False
            in_green = False
        
        if not has_hit_red_green:
            if in_red:
                sim_data['rg_outcome'] = 'red'
                has_hit_red_green = True
                sim_data['rg_hit_timestep'] = frame
            elif in_green:
                sim_data['rg_outcome'] = 'green'
                has_hit_red_green = True
                sim_data['rg_hit_timestep'] = frame
        
        if has_hit_red_green:
            break
    
    sim_data['num_frames'] = frame+1 # this cannot be frames, because ball may hit red or green before the 
    
    # Process distractors if provided
    if distractorParams:
        if not DISTRACTOR_FUNCTIONS_AVAILABLE:
            pretty_warning("Distractor parameters provided but distractor functions not available. Skipping distractors.")
        else:
            print("Processing distractors...")
            
            # Process key distractors
            keyDistractors = distractorParams.get('keyDistractors', [])
            for i, keyDistractor in enumerate(keyDistractors):
                print(f"Processing key distractor {i+1}/{len(keyDistractors)}")
                distractor_data = simulate_key_distractor(
                    keyDistractor, 
                    sim_data, 
                    worldWidth, 
                    worldHeight,
                    TIMESTEP,
                    FRAME_INTERVAL,
                    FPS,
                    ballSpeed,
                    elasticity,
                    friction,
                    space  # Reuse the space for collision detection with barriers
                )
                sim_data['key_distractors'].append(distractor_data)
            
            # Process random distractors
            randomParams = distractorParams.get('randomDistractorParams', {})
            if randomParams and randomParams.get('probability', 0) > 0:
                print("Generating random distractors...")
                random_distractors = generate_random_distractors(
                    randomParams,
                    sim_data,
                    worldWidth,
                    worldHeight,
                    TIMESTEP,
                    FRAME_INTERVAL,
                    FPS,
                    ballSpeed,
                    elasticity,
                    friction,
                    space
                )
                sim_data['random_distractors'] = random_distractors
    
    return sim_data

def load_init_state_entities(folder_path: Path, json_filename: str = "init_state_entities.json") -> Dict:
    """Load init_state_entities.json (or custom filename) from the folder."""
    init_file = folder_path / json_filename
    if not init_file.exists():
        raise FileNotFoundError(f"{json_filename} not found in {folder_path}")
    
    with open(init_file, 'r') as f:
        data = json.load(f)
    
    return data

def prepare_simulation_params(simulationParams: Dict) -> tuple:
    """
    Extract and prepare simulation parameters.
    
    Expected keys in simulationParams:
    - videoLength
    - ballSpeed
    - fps
    - physicsStepsPerFrame (optional, defaults to 12)
    - res_multiplier (optional, defaults to 4)
    - timestep (optional, defaults to 0.01)
    - worldWidth
    - worldHeight
    """
    videoLength = simulationParams['videoLength']
    ballSpeed = simulationParams['ballSpeed']
    fps = simulationParams['fps']
    physicsStepsPerFrame = simulationParams.get('physicsStepsPerFrame', 12)
    res_multiplier = simulationParams.get('res_multiplier', 4)
    timestep = simulationParams.get('timestep', 0.01)
    worldWidth = simulationParams['worldWidth']
    worldHeight = simulationParams['worldHeight']
    
    return (videoLength, ballSpeed, fps, physicsStepsPerFrame, res_multiplier, timestep, worldWidth, worldHeight)

def run_simulation(folder_path: Path, json_filename: str = "init_state_entities.json") -> Dict:
    """
    Run simulation for a single folder.
    
    Args:
        folder_path: Path to folder containing the JSON file
        json_filename: Name of the JSON file to load (default: "init_state_entities.json")
    
    Returns:
        Simulation data dictionary
    """
    pretty_rule("SIMULATION RUN")
    pretty_info(f"Loading {json_filename} from:\n  {folder_path}")
    
    # Load init state entities
    init_data = load_init_state_entities(folder_path, json_filename)
    entities = init_data['entities']
    simulationParams = init_data['simulationParams']
    distractorParams = init_data.get('distractorParams', None)
    
    pretty_info(f"Found {len(entities)} entities")
    if distractorParams:
        pretty_info("Distractor parameters found")
    else:
        pretty_info("No distractor parameters")
    
    # Prepare simulation parameters
    sim_params = prepare_simulation_params(simulationParams)
    
    # Run simulation
    pretty_info("Running physics simulation...")
    start_time = time.time()
    sim_data = run_simulation_with_visualization(entities, sim_params, distractorParams)
    end_time = time.time()
    
    pretty_success(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Save simulation data
    output_file = folder_path / "simulation_data.json"
    pretty_info(f"Saving simulation data to:\n  {output_file}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    sim_data_serializable = convert_to_json_serializable(sim_data)
    
    with open(output_file, 'w') as f:
        json.dump(sim_data_serializable, f, indent=2)
    
    pretty_success(Panel(f"Simulation data saved to:\n{output_file}", title="Saved"))
    
    # Generate WebM videos
    try:
        generate_webm_videos(sim_data, folder_path)
    except Exception as e:
        pretty_warning(f"Video generation failed: {e}")
        pretty_warning("Continuing without videos...")
    
    # Print summary
    pretty_rule("SIMULATION SUMMARY")
    summary = {
        "Total frames": sim_data['num_frames'],
        "FPS": sim_data['fps'],
        "Total time (seconds)": sim_data['num_frames'] / sim_data['fps'],
        "Red/Green outcome": sim_data.get('rg_outcome', 'None'),
        "Number of barriers": sim_data['num_barriers'],
        "Number of occluders": sim_data['num_occs'],
    }
    if distractorParams:
        summary["Key distractors"] = len(sim_data.get('key_distractors', []))
        summary["Random distractors"] = len(sim_data.get('random_distractors', []))
    
    for key, value in summary.items():
        pretty_info(f"{key}: {value}")
    
    return sim_data

def run_batch_simulations(
    folder_path: Path,
    json_filename: str = "init_state_entities.json"
) -> Dict[str, Dict]:
    """
    Run simulations for all subfolders in a parent folder.
    
    Args:
        folder_path: Path to folder containing multiple trial folders
        json_filename: Name of the JSON file to load from each subfolder (default: "init_state_entities.json")
    
    Returns:
        Dictionary mapping trial names to simulation data
    """
    pretty_rule("BATCH SIMULATION PROCESSING")
    pretty_info(f"Scanning folder for trial subfolders:\n  {folder_path}")
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Get all subdirectories
    trial_folders = [f for f in folder_path.iterdir() if f.is_dir()]
    trial_folders.sort()
    
    if not trial_folders:
        pretty_error(f"No trial folders found in: {folder_path}")
        raise ValueError(f"No trial folders found in: {folder_path}")
    
    pretty_info(f"Found {len(trial_folders)} trial folders to process.")
    
    experiment_results = {}
    
    # Use rich progress bar if available
    if console is not None:
        trial_iter = track(trial_folders, description="Processing trials", transient=True)
    else:
        trial_iter = tqdm(trial_folders, desc="Processing trials")
    
    for trial_folder in trial_iter:
        trial_name = trial_folder.name
        try:
            pretty_info(f"\nProcessing trial: {trial_name}")
            sim_data = run_simulation(trial_folder, json_filename)
            experiment_results[trial_name] = sim_data
            pretty_success(f"Trial {trial_name} completed successfully!")
        except Exception as e:
            pretty_error(f"Error processing trial {trial_name}: {e}")
            continue
    
    if experiment_results:
        pretty_rule("BATCH PROCESSING SUMMARY")
        pretty_info(f"Successfully processed {len(experiment_results)} out of {len(trial_folders)} trials.")
        
        # Print summary statistics
        total_frames = sum(sim_data.get('num_frames', 0) for sim_data in experiment_results.values())
        avg_frames = total_frames / len(experiment_results) if experiment_results else 0
        
        summary = {
            "Total trials processed": len(experiment_results),
            "Total frames (all trials)": total_frames,
            "Average frames per trial": f"{avg_frames:.1f}",
        }
        
        for key, value in summary.items():
            pretty_info(f"{key}: {value}")
    else:
        pretty_error("No trials were processed successfully.")
        raise ValueError("No trials were processed successfully.")
    
    return experiment_results

def render_frame(
    sim_data: Dict,
    frame_index: int,
    canvas_width: int,
    canvas_height: int,
    world_width: float,
    world_height: float,
    disguise_distractors: bool = False,
    lift_up_target: bool = False,
    scale_factor: int = 3
) -> np.ndarray:
    """
    Render a single frame of the simulation.
    
    Args:
        sim_data: Simulation data dictionary
        frame_index: Frame index to render
        canvas_width: Width of canvas in pixels
        canvas_height: Height of canvas in pixels
        world_width: World width in simulation units
        world_height: World height in simulation units
        disguise_distractors: If True, render distractors as blue (disguised)
        lift_up_target: If True, render target above occluders
        scale_factor: Scale factor for high-resolution rendering
    
    Returns:
        numpy array representing the frame (BGR format for OpenCV)
    """
    # Create high-resolution canvas
    hd_width = canvas_width * scale_factor
    hd_height = canvas_height * scale_factor
    
    # Create white background
    frame = np.ones((hd_height, hd_width, 3), dtype=np.uint8) * 255
    
    # Helper function to convert world coordinates to canvas coordinates
    def world_to_canvas(wx: float, wy: float) -> Tuple[int, int]:
        cx = int((wx / world_width) * hd_width)
        cy = int(hd_height - ((wy / world_height) * hd_height))
        return cx, cy
    
    # Render green sensor first
    if 'green_sensor' in sim_data:
        sensor = sim_data['green_sensor']
        x1, y1 = world_to_canvas(sensor['x'], sensor['y'])
        x2, y2 = world_to_canvas(sensor['x'] + sensor['width'], sensor['y'] + sensor['height'])
        cv2.rectangle(frame, (x1, y2), (x2, y1), (0, 255, 0), -1)
    
    # Render red sensor
    if 'red_sensor' in sim_data:
        sensor = sim_data['red_sensor']
        x1, y1 = world_to_canvas(sensor['x'], sensor['y'])
        x2, y2 = world_to_canvas(sensor['x'] + sensor['width'], sensor['y'] + sensor['height'])
        cv2.rectangle(frame, (x1, y2), (x2, y1), (0, 0, 255), -1)
    
    # Render barriers
    for barrier in sim_data.get('barriers', []):
        x1, y1 = world_to_canvas(barrier['x'], barrier['y'])
        x2, y2 = world_to_canvas(barrier['x'] + barrier['width'], barrier['y'] + barrier['height'])
        cv2.rectangle(frame, (x1, y2), (x2, y1), (0, 0, 0), -1)
    
    # Helper function to render target
    def render_target(is_lifted: bool = False):
        if 'step_data' in sim_data and frame_index in sim_data['step_data']:
            target_data = sim_data['step_data'][frame_index]
            target_size = sim_data['target']['size']
            radius = target_size / 2
            tx = target_data['x']
            ty = target_data['y']
            
            # Convert center position to canvas coordinates
            center_wx = tx + radius
            center_wy = ty + radius
            cx, cy = world_to_canvas(center_wx, center_wy)
            canvas_radius = int(radius * (hd_width / world_width))
            
            # Check if under occluder when lifted
            if is_lifted:
                under_occluder = False
                for occluder in sim_data.get('occluders', []):
                    if (tx + radius > occluder['x'] and tx < occluder['x'] + occluder['width'] and
                        ty + radius > occluder['y'] and ty < occluder['y'] + occluder['height']):
                        under_occluder = True
                        break
                
                if under_occluder:
                    # Blend blue with gray occluder at 30% opacity: blue * 0.3 + gray * 0.7
                    # Gray is (128, 128, 128) in BGR, Blue is (255, 0, 0) in BGR
                    color = (
                        int(255 * 0.3 + 128 * 0.7),  # B
                        int(0 * 0.3 + 128 * 0.7),   # G
                        int(0 * 0.3 + 128 * 0.7)    # R
                    )
                else:
                    color = (255, 0, 0)  # Blue (BGR)
            else:
                color = (255, 0, 0)  # Blue (BGR)
            
            cv2.circle(frame, (cx, cy), canvas_radius, color, -1)
            
            # Add border if lifted
            if lift_up_target and is_lifted:
                cv2.circle(frame, (cx, cy), canvas_radius, (0, 0, 0), 2 * scale_factor)
    
    # Render distractors before occluders if not lifted and revealed
    if not (lift_up_target and not disguise_distractors):
        # Render key distractors
        for distractor in sim_data.get('key_distractors', []):
            if 'step_data' in distractor and frame_index in distractor['step_data']:
                hal_data = distractor['step_data'][frame_index]
                target_size = sim_data['target']['size']
                radius = target_size / 2
                tx = hal_data['x']
                ty = hal_data['y']
                
                center_wx = tx + radius
                center_wy = ty + radius
                cx, cy = world_to_canvas(center_wx, center_wy)
                canvas_radius = int(radius * (hd_width / world_width))
                
                color = (255, 0, 0) if disguise_distractors else (226, 43, 138)  # Blue or Purple (BGR)
                cv2.circle(frame, (cx, cy), canvas_radius, color, -1)
        
        # Render random distractors
        for distractor in sim_data.get('random_distractors', []):
            if 'step_data' in distractor and frame_index in distractor['step_data']:
                hal_data = distractor['step_data'][frame_index]
                target_size = sim_data['target']['size']
                radius = target_size / 2
                tx = hal_data['x']
                ty = hal_data['y']
                
                center_wx = tx + radius
                center_wy = ty + radius
                cx, cy = world_to_canvas(center_wx, center_wy)
                canvas_radius = int(radius * (hd_width / world_width))
                
                color = (255, 0, 0) if disguise_distractors else (180, 105, 255)  # Blue or Pink (BGR)
                cv2.circle(frame, (cx, cy), canvas_radius, color, -1)
    
    # Render target before occluders if not lifted
    if not lift_up_target:
        render_target(False)
    
    # Render occluders
    for occluder in sim_data.get('occluders', []):
        x1, y1 = world_to_canvas(occluder['x'], occluder['y'])
        x2, y2 = world_to_canvas(occluder['x'] + occluder['width'], occluder['y'] + occluder['height'])
        cv2.rectangle(frame, (x1, y2), (x2, y1), (128, 128, 128), -1)
    
    # Render target after occluders if lifted
    if lift_up_target:
        render_target(True)
    
    # Render distractors after occluders if lifted and revealed
    if lift_up_target and not disguise_distractors:
        # Render key distractors
        for distractor in sim_data.get('key_distractors', []):
            if 'step_data' in distractor and frame_index in distractor['step_data']:
                hal_data = distractor['step_data'][frame_index]
                target_size = sim_data['target']['size']
                radius = target_size / 2
                tx = hal_data['x']
                ty = hal_data['y']
                
                center_wx = tx + radius
                center_wy = ty + radius
                cx, cy = world_to_canvas(center_wx, center_wy)
                canvas_radius = int(radius * (hd_width / world_width))
                
                # Check if under occluder
                under_occluder = False
                for occluder in sim_data.get('occluders', []):
                    if (tx + radius > occluder['x'] and tx < occluder['x'] + occluder['width'] and
                        ty + radius > occluder['y'] and ty < occluder['y'] + occluder['height']):
                        under_occluder = True
                        break
                
                if under_occluder:
                    # Blend purple with gray occluder at 30% opacity
                    # Purple is (226, 43, 138) in BGR, Gray is (128, 128, 128) in BGR
                    color = (
                        int(226 * 0.3 + 128 * 0.7),  # B
                        int(43 * 0.3 + 128 * 0.7),   # G
                        int(138 * 0.3 + 128 * 0.7)   # R
                    )
                else:
                    color = (226, 43, 138)  # Purple (BGR)
                
                cv2.circle(frame, (cx, cy), canvas_radius, color, -1)
                cv2.circle(frame, (cx, cy), canvas_radius, (0, 0, 0), 2 * scale_factor)  # Border
        
        # Render random distractors
        for distractor in sim_data.get('random_distractors', []):
            if 'step_data' in distractor and frame_index in distractor['step_data']:
                hal_data = distractor['step_data'][frame_index]
                target_size = sim_data['target']['size']
                radius = target_size / 2
                tx = hal_data['x']
                ty = hal_data['y']
                
                center_wx = tx + radius
                center_wy = ty + radius
                cx, cy = world_to_canvas(center_wx, center_wy)
                canvas_radius = int(radius * (hd_width / world_width))
                
                # Check if under occluder
                under_occluder = False
                for occluder in sim_data.get('occluders', []):
                    if (tx + radius > occluder['x'] and tx < occluder['x'] + occluder['width'] and
                        ty + radius > occluder['y'] and ty < occluder['y'] + occluder['height']):
                        under_occluder = True
                        break
                
                if under_occluder:
                    # Blend pink with gray occluder at 30% opacity
                    # Pink is (180, 105, 255) in BGR, Gray is (128, 128, 128) in BGR
                    color = (
                        int(180 * 0.3 + 128 * 0.7),  # B
                        int(105 * 0.3 + 128 * 0.7),  # G
                        int(255 * 0.3 + 128 * 0.7)   # R
                    )
                else:
                    color = (180, 105, 255)  # Pink (BGR)
                
                cv2.circle(frame, (cx, cy), canvas_radius, color, -1)
                cv2.circle(frame, (cx, cy), canvas_radius, (0, 0, 0), 2 * scale_factor)  # Border
    
    return frame

def generate_webm_video(
    sim_data: Dict,
    output_path: Path,
    fps: float,
    disguise_distractors: bool = False,
    lift_up_target: bool = False
) -> None:
    """
    Generate a WebM video from simulation data.
    
    Args:
        sim_data: Simulation data dictionary
        output_path: Path to save the video file
        fps: Frames per second
        disguise_distractors: If True, render distractors as blue (disguised)
        lift_up_target: If True, render target above occluders
    """
    world_width, world_height = sim_data['scene_dims']
    interval = sim_data['interval']
    num_frames = sim_data['num_frames']
    
    # Calculate canvas dimensions (same as JavaScript)
    canvas_width = int(world_width / interval)
    canvas_height = int(world_height / interval)
    
    # Scale factor for high quality (3x like JavaScript)
    scale_factor = 3
    hd_width = canvas_width * scale_factor
    hd_height = canvas_height * scale_factor
    
    # Try to use imageio for WebM (more reliable than OpenCV)
    try:
        import imageio
        imageio_available = True
    except ImportError:
        imageio_available = False
    
    if imageio_available:
        # Use imageio for WebM (requires ffmpeg)
        try:
            # Set ffmpeg parameters to suppress warnings and set quality explicitly
            ffmpeg_params = [
                '-loglevel', 'error',  # Only show errors, suppress warnings
                '-crf', '32',  # Set CRF explicitly to avoid the warning
            ]
            
            writer = imageio.get_writer(
                str(output_path),
                fps=fps,
                codec='libvpx-vp9',
                quality=8,
                pixelformat='yuv420p',
                ffmpeg_params=ffmpeg_params
            )
            
            # Render and write each frame
            for frame_index in tqdm(range(num_frames), desc=f"Rendering {output_path.name}", leave=False):
                frame = render_frame(
                    sim_data,
                    frame_index,
                    canvas_width,
                    canvas_height,
                    world_width,
                    world_height,
                    disguise_distractors=disguise_distractors,
                    lift_up_target=lift_up_target,
                    scale_factor=scale_factor
                )
                # Convert BGR to RGB for imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
            
            writer.close()
            return
        except Exception as e:
            pretty_warning(f"imageio failed, trying OpenCV: {e}")
    
    # Fallback to OpenCV (may not support WebM, will try different formats)
    # Try VP9 codec first
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    if fourcc == -1:
        fourcc = cv2.VideoWriter_fourcc(*'vp90')
    if fourcc == -1:
        # Try VP8
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
    if fourcc == -1:
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
    
    # If WebM codecs don't work, try MP4 as fallback
    if fourcc == -1:
        pretty_warning("WebM codec not available, trying MP4 format")
        output_path = output_path.with_suffix('.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if fourcc == -1:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (hd_width, hd_height)
    )
    
    if not out.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {output_path}. "
            "Try installing imageio and ffmpeg: pip install imageio[ffmpeg]"
        )
    
    # Render and write each frame
    for frame_index in tqdm(range(num_frames), desc=f"Rendering {output_path.name}", leave=False):
        frame = render_frame(
            sim_data,
            frame_index,
            canvas_width,
            canvas_height,
            world_width,
            world_height,
            disguise_distractors=disguise_distractors,
            lift_up_target=lift_up_target,
            scale_factor=scale_factor
        )
        out.write(frame)
    
    out.release()

def generate_webm_videos(sim_data: Dict, folder_path: Path) -> None:
    """
    Generate stimulus and revealed WebM videos.
    
    Args:
        sim_data: Simulation data dictionary
        folder_path: Path to folder where videos should be saved
    """
    trial_name = folder_path.name
    fps = sim_data['fps']
    
    pretty_info("Generating WebM videos...")
    
    # Generate stimulus video (disguise_distractors=True, lift_up_target=False)
    stimulus_path = folder_path / f"{trial_name}_stimulus.webm"
    pretty_info(f"Generating stimulus video: {stimulus_path.name}")
    try:
        generate_webm_video(
            sim_data,
            stimulus_path,
            fps,
            disguise_distractors=True,
            lift_up_target=False
        )
        pretty_success(f"Stimulus video saved to: {stimulus_path}")
    except Exception as e:
        pretty_error(f"Failed to generate stimulus video: {e}")
        raise
    
    # Generate revealed video (disguise_distractors=False, lift_up_target=True)
    revealed_path = folder_path / f"{trial_name}_revealed.webm"
    pretty_info(f"Generating revealed video: {revealed_path.name}")
    try:
        generate_webm_video(
            sim_data,
            revealed_path,
            fps,
            disguise_distractors=False,
            lift_up_target=True
        )
        pretty_success(f"Revealed video saved to: {revealed_path}")
    except Exception as e:
        pretty_error(f"Failed to generate revealed video: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run physics simulation from init_state_entities.json")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stimulus_path", type=str, 
                       help="Path to single trial folder containing the JSON file")
    group.add_argument("--stimulus_folder", type=str, 
                       help="Path to folder containing multiple trial folders (for batch processing)")
    
    parser.add_argument("--json_filename", type=str, default="init_state_entities.json",
                       help="Name of the JSON file to load (default: init_state_entities.json)")
    parser.add_argument("--batch_mode", action="store_true",
                       help="Enable batch processing mode (required when using --stimulus_folder)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stimulus_folder and not args.batch_mode:
        parser.error("--batch_mode is required when using --stimulus_folder")
    if args.stimulus_path and args.batch_mode:
        parser.error("--batch_mode cannot be used with --stimulus_path")
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    try:
        if args.stimulus_path:
            # Single trial mode
            folder_path = Path(args.stimulus_path)
            if not folder_path.is_absolute():
                folder_path = project_root / args.stimulus_path
            
            if not folder_path.exists():
                pretty_error(f"Folder not found: {folder_path}")
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            if not folder_path.is_dir():
                pretty_error(f"Path is not a directory: {folder_path}")
                raise ValueError(f"Path is not a directory: {folder_path}")
            
            sim_data = run_simulation(folder_path, args.json_filename)
            pretty_success("Simulation completed successfully!")
            
        else:
            # Batch processing mode
            folder_folder = Path(args.stimulus_folder)
            if not folder_folder.is_absolute():
                folder_folder = project_root / args.stimulus_folder
            
            if not folder_folder.exists():
                pretty_error(f"Folder not found: {folder_folder}")
                raise FileNotFoundError(f"Folder not found: {folder_folder}")
            
            if not folder_folder.is_dir():
                pretty_error(f"Path is not a directory: {folder_folder}")
                raise ValueError(f"Path is not a directory: {folder_folder}")
            
            results = run_batch_simulations(folder_folder, args.json_filename)
            pretty_success(f"Batch processing completed! Processed [bold]{len(results)}[/bold] trials.")
            
    except Exception as e:
        pretty_error(f"Error during simulation: {e}")
        raise

if __name__ == "__main__":
    main()

