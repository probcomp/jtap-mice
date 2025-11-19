# Standard library imports
from functools import partial

# Third-party imports
import json
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.debug import print as jprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm

# GenJAX imports
import genjax
from genjax import gen, Pytree, ExactDensity, ChoiceMapBuilder as C

# JTAP imports
from jtap_mice.core import SuperPytree
from jtap_mice.distributions import uniformcat, labcat
from jtap_mice.utils import init_step_concat


from IPython.display import HTML as HTML_Display

# ALL VIZ FUNCTIONS IN THIS FILE ARE FOR NOTEBOOKS VIZ HELPERS

# Note: this is a continuous model

@Pytree.dataclass
class LR_Hyperparams(SuperPytree):
    # Assume scene height is 1
    max_speed: jnp.ndarray = SuperPytree.field()
    diameter: jnp.ndarray = SuperPytree.field()
    left_right_labels: jnp.ndarray = SuperPytree.field()
    masked_occluders: jnp.ndarray = SuperPytree.field() # N by 2 array of [x, length] occluder values
    pix_x: jnp.ndarray = SuperPytree.field()
    pix_y: jnp.ndarray = SuperPytree.field()
    pos_noise_std: jnp.ndarray = SuperPytree.field()
    speed_noise_std: jnp.ndarray = SuperPytree.field()
    direction_flip_prob: jnp.ndarray = SuperPytree.field()
    scene_length: jnp.ndarray = SuperPytree.static() # length 0 is the left edge, length scene_length is the right edge
    pixel_density: jnp.ndarray = SuperPytree.static()
    num_occluders: jnp.ndarray = SuperPytree.static()
    is_semi_markov_switching: jnp.bool_ = SuperPytree.static()

    def __eq__(self, other) -> bool:
        if jax.tree_util.tree_structure(self) != jax.tree_util.tree_structure(other):
            return False
        leaves1 = jax.tree_util.tree_leaves(self)
        leaves2 = jax.tree_util.tree_leaves(other)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))
    
    @classmethod
    def create(cls, **kwargs):
        scene_length = kwargs['scene_length']
        pixel_density = kwargs['pixel_density']

        pix_x = jnp.array(np.linspace(0, scene_length, (pixel_density * scene_length).astype(np.int32), endpoint=False))
        pix_y = jnp.array(np.linspace(0, 1, (pixel_density).astype(np.int32), endpoint=False))

        num_occluders = kwargs['masked_occluders'].shape[0]

        return cls(
            **kwargs,
            pix_x=pix_x,
            pix_y=pix_y,
            num_occluders=num_occluders,
        )
        

@Pytree.dataclass
class LR_State(SuperPytree):
    hypers: LR_Hyperparams 
    x: jnp.ndarray
    speed: jnp.ndarray
    direction: jnp.ndarray
    hit_boundary: jnp.ndarray
    tstep : jnp.ndarray
    
@Pytree.dataclass
class LR_RenderArgs(SuperPytree):
    x: jnp.ndarray = SuperPytree.field()
    diameter: jnp.ndarray = SuperPytree.field()
    masked_occluders: jnp.ndarray = SuperPytree.field()
    pix_x: jnp.ndarray = SuperPytree.field()
    pix_y: jnp.ndarray = SuperPytree.field()
    num_occluders: jnp.ndarray = SuperPytree.static()
    pixel_density: jnp.ndarray = SuperPytree.static()


@gen
def lr_init_model(hypers: LR_Hyperparams):

    max_speed = hypers.max_speed
    scene_length = hypers.scene_length
    diameter = hypers.diameter
    left_right_labels = hypers.left_right_labels
    speed = genjax.uniform(jnp.float32(0.), max_speed) @ "speed"
    direction = uniformcat(left_right_labels) @ "direction" # -1 is left, 1 is right

    x = genjax.uniform(jnp.float32(0.), scene_length - diameter) @ "x"

    hit_boundary = (x <= 0) | (x >= scene_length - diameter)

    return LR_State(
        tstep=jnp.array(0),
        hypers=hypers,
        x=x,
        speed=speed,
        direction=direction,
        hit_boundary=hit_boundary
    )
    
lr_init_simulate = lr_init_model.simulate
lr_init_jsimulate = jax.jit(lr_init_model.simulate)
lr_init_importance = lr_init_model.importance
lr_init_jimportance = jax.jit(lr_init_model.importance)

@gen
def lr_stepper_model(state: LR_State, switch_time_array: jnp.ndarray):
    # NOTE: need to discuss: what is the causal model here?

    epsilon = jnp.float32(1e-5)

    x = state.x
    speed = state.speed
    direction = state.direction
    max_speed = state.hypers.max_speed
    diameter = state.hypers.diameter
    scene_length = state.hypers.scene_length
    pos_noise_std = state.hypers.pos_noise_std
    speed_noise_std = state.hypers.speed_noise_std
    direction_flip_prob = state.hypers.direction_flip_prob

    # this one line is the "physics" of the model
    x_mean = x + (speed * direction)

    # we have to make sure that the ball does not go out of bounds. If it does, then we record it in the state and clip it to the boundary
    x_mean = jnp.clip(x_mean, epsilon, scene_length - diameter - epsilon)

    # then we sample the position from the truncated normal distribution
    x = genjax.truncated_normal(x_mean, pos_noise_std, jnp.float32(0.), scene_length - diameter) @ "x"
    speed = genjax.truncated_normal(speed, speed_noise_std, jnp.float32(0.), max_speed) @ "speed"

    # if semi-Markov switching is enabled, then we use the switch time array to determine the direction flip probability (force it to 0 or 1). This is pre-computed.
    direction_flip_prob = jnp.where(state.hypers.is_semi_markov_switching, switch_time_array[state.tstep], direction_flip_prob)

    direction = direction_flip_distribution(direction, direction_flip_prob) @ "direction"


    hit_boundary = (x <= (0 + epsilon)) | (x >= (scene_length - diameter - epsilon))

    return LR_State(
        hypers=state.hypers,
        tstep=state.tstep + 1,
        x=x,
        speed=speed,
        direction=direction,
        hit_boundary=hit_boundary
    )

lr_stepper_simulate = lr_stepper_model.simulate
lr_stepper_jsimulate = jax.jit(lr_stepper_model.simulate)
lr_stepper_importance = lr_stepper_model.importance
lr_stepper_jimportance = jax.jit(lr_stepper_model.importance)

@jax.jit
def render_lr_scene(lr_render_args: LR_RenderArgs):
    """
    Render a 1D left-right scene with height=1.
    
    Args:
        x: position of the ball's left edge
        diameter: diameter of the ball
        masked_occluders: N by 2 array of [x_start, length] occluder values
        scene_length: total length of the scene
        pixel_density: pixels per unit length
    
    Returns:
        image: 2D array with shape (pixel_density, scene_length * pixel_density) where 0=background, 1=occluder, 2=target
    """
    # Create pixel grid for the scene

    x = lr_render_args.x
    diameter = lr_render_args.diameter
    num_occluders = lr_render_args.num_occluders
    masked_occluders = lr_render_args.masked_occluders
    pixel_density = lr_render_args.pixel_density
    pix_x = lr_render_args.pix_x
    pix_y = lr_render_args.pix_y

    num_pixels_x = pix_x.shape[0]
    num_pixels_y = pix_y.shape[0]

    # Create 2D meshgrid
    y_vals, x_vals = jnp.meshgrid(pix_y, pix_x, indexing='ij')
    
    # Initialize image with background (0)
    image = jnp.zeros((num_pixels_y, num_pixels_x), dtype=jnp.int8)
    
    r = diameter / 2
    # Calculate center coordinates by adding radius to position
    target_center_x_spatial = x + r
    target_center_y_spatial = r # center of ball is 0.5 always
    
    # Snap center to closest values in pix_x and pix_y arrays
    target_center_x = pix_x[jnp.argmin(jnp.abs(pix_x - target_center_x_spatial))]
    target_center_y = pix_y[jnp.argmin(jnp.abs(pix_y - target_center_y_spatial))]
    
    # Use a small epsilon to handle floating point precision issues
    distance_squared = (x_vals - target_center_x + 0.5*(1/pixel_density))**2 + (y_vals - target_center_y + 0.5*(1/pixel_density))**2
    r_squared = r**2
    
    image = jnp.where((distance_squared) < (r_squared), jnp.int8(2), image)

    # Render occluders (value 1)
    for i in range(num_occluders):
        occluder_x, occluder_length = masked_occluders[i]
        occluder_mask = (x_vals >= occluder_x) & (x_vals < occluder_x + occluder_length)
        image = jnp.where(occluder_mask, jnp.int8(1), image)

    return image

def get_lr_render_args(state: LR_State):
    return LR_RenderArgs(
        x=state.x,
        diameter=state.hypers.diameter,
        num_occluders=state.hypers.num_occluders,
        masked_occluders=state.hypers.masked_occluders,
        pixel_density=state.hypers.pixel_density,
        pix_x=state.hypers.pix_x,
        pix_y=state.hypers.pix_y
    )

@Pytree.dataclass
class DirectionFlipDistribution(ExactDensity):
    def sample(self, key, direction, direction_flip_prob, **kwargs):
        flip = jax.random.bernoulli(key, p = direction_flip_prob) * direction
        return jnp.where(flip, -direction, direction)

    def logpdf(self, x, direction, direction_flip_prob, **kwargs):
        same = (x == direction)
        flipped = (x == -direction)
        return jnp.where(
            same, jnp.log1p(-direction_flip_prob),  # log(1 - p)
            jnp.where(flipped, jnp.log(direction_flip_prob), -jnp.inf)
        )
    
direction_flip_distribution = DirectionFlipDistribution()

@Pytree.dataclass
class LR_PixelFlipLikelihood(ExactDensity):
    # HARDCODED TO HAVE 6 POSSIBLE PIXEL VALUES
    def sample(self, key, lr_render_args, flip_prob, *args, **kwargs):
        rendered_image = render_lr_scene(lr_render_args)
        
        # Apply pixel flip noise
        flip_key, noise_key = jax.random.split(key)
        
        # Determine which pixels to flip based on flip_prob
        flip_mask = jax.random.uniform(flip_key, rendered_image.shape) < flip_prob        
        
        # For each pixel, create a mask of valid alternatives (all values except the original)
        def sample_alternative_pixel(original_val, rng_key):
            # Instead of boolean indexing, use jnp.where to select alternatives
            # This avoids the NonConcreteBooleanIndexError
            alternatives = jnp.array([0, 1, 2], dtype=jnp.int8)
            # Remove the original value by setting its probability to 0
            probs = jnp.where(alternatives == original_val, 0.0, 1.0)
            # Normalize probabilities
            probs = probs / jnp.sum(probs)
            # Sample using categorical distribution
            return jax.random.choice(rng_key, alternatives, p=probs)
        
        # Generate noise keys for each pixel
        noise_keys = jax.random.split(noise_key, rendered_image.size)
        noise_keys = noise_keys.reshape(rendered_image.shape + (2,))
        
        # Apply pixel flipping
        flipped_values = jax.vmap(jax.vmap(sample_alternative_pixel))(
            rendered_image, noise_keys
        )
        
        # Return original where not flipped, alternative where flipped
        return jnp.where(flip_mask, flipped_values, rendered_image)

    def logpdf(self, obs_image, render_args, flip_prob, *args, **kwargs):
        return jnp.sum(
                # jnp.where(obs_image == render_scene(*render_args), jnp.log(1 - flip_prob), jnp.log(flip_prob/5))
                jnp.where(obs_image == render_scene(*render_args), jnp.log(1 - flip_prob), jnp.log(flip_prob))
        )
    
lr_pixel_flip_likelihood = LR_PixelFlipLikelihood()

# Define function to create trial indices by bin
def create_trial_indices_by_bin(num_bins, starting_positions, same_side_array, range_start, bin_length):
    """
    Create a dictionary mapping bin indices to trial indices, separated by same/different side outcomes.
    
    Args:
        num_bins: Number of bins to create
        starting_positions: Array of starting positions for all trials
        same_side_array: Boolean array indicating if trial ended on same side as it started
        range_start: Starting position of the uniform range
        bin_length: Length of each bin
    
    Returns:
        Dictionary with bin indices as keys, each containing 'same_side' and 'different_side' trial lists
    """
    trial_indices_dict = {}
    
    for bin_idx in range(num_bins):
        bin_start = range_start + bin_idx * bin_length
        bin_end = bin_start + bin_length
        
        # Find trials that fall within this bin
        in_bin_mask = (starting_positions >= bin_start) & (starting_positions < bin_end)
        trials_in_bin = np.where(in_bin_mask)[0]
        
        # Separate by same/different side outcomes
        same_side_trials = trials_in_bin[same_side_array[trials_in_bin]]
        different_side_trials = trials_in_bin[~same_side_array[trials_in_bin]]
        
        trial_indices_dict[bin_idx] = {
            'same_side': same_side_trials.tolist(),
            'different_side': different_side_trials.tolist()
        }
    
    return trial_indices_dict

def render_lr_frame(x_pos, diameter, scene_length, pixel_density):
    """
    Render a 1D left-right scene with height=1.
    
    Args:
        x_pos: position of the ball's center
        diameter: diameter of the ball
        scene_length: total length of the scene
        pixel_density: pixels per unit length (per diameter)
    
    Returns:
        image: 2D array with shape (pixel_density, scene_length * pixel_density) where 0=background, 1=occluder, 2=target
    """
    # Create pixel grid for the scene

    pixel_density = np.int32(pixel_density)
    scene_length = np.float32(scene_length)
    pix_x = np.array(np.linspace(0, scene_length, (pixel_density * scene_length).astype(np.int32), endpoint=False))
    pix_y = np.array(np.linspace(0, 1, (pixel_density).astype(np.int32), endpoint=False))

    num_pixels_x = pix_x.shape[0]
    num_pixels_y = pix_y.shape[0]

    # Create 2D meshgrid
    y_vals, x_vals = np.meshgrid(pix_y, pix_x, indexing='ij')
    
    # Initialize image with background (0)
    image = np.ones((num_pixels_y, num_pixels_x, 3), dtype=np.uint8) * 255
    
    r = diameter / 2
    # Calculate center coordinates by adding radius to position
    target_center_x_spatial = x_pos
    target_center_y_spatial = r # center of ball is 0.5 always
    
    # Snap center to closest values in pix_x and pix_y arrays
    target_center_x = pix_x[np.argmin(np.abs(pix_x - target_center_x_spatial))]
    target_center_y = pix_y[np.argmin(np.abs(pix_y - target_center_y_spatial))]
    
    # Use a small epsilon to handle floating point precision issues
    distance_squared = (x_vals - target_center_x + 0.5*(1/pixel_density))**2 + (y_vals - target_center_y + 0.5*(1/pixel_density))**2
    r_squared = r**2
    
    # Create mask for pixels within the circle
    circle_mask = (distance_squared) < (r_squared)
    
    # Set pixels within the circle to blue (0, 0, 255)
    image[circle_mask] = [0, 0, 255]

    return image

def load_trial_data(json_path):
    """
    Load trial data from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing trial data
        
    Returns:
        dict: Dictionary containing 'config' and 'trial_data' keys
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert trial_data keys back to integers and values back to numpy arrays
    trial_data = {int(k): np.array(v) for k, v in data['trial_data'].items()}
    
    return {
        'config': data['config'],
        'trial_data': trial_data
    }



def display_video(frames, framerate=30, skip_t = 1, max_figsize_width = 6):
    """
    frames: list of N np.arrays (H x W x 3)
    framerate: frames per second
    """
    height, width, _ = frames[0].shape
    dpi = 70

    num_frames = len(frames)
    if skip_t != 1:
      frames = frames[::skip_t]
    # orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.

    fig, ax = plt.subplots(1, 1, figsize=(max_figsize_width, max_figsize_width*(height/width)))
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    # ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frames[frame])
        # return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(frames.shape[0]),
      interval=interval, blit=False, repeat=True)
    plt.close()
    return HTML_Display(anim.to_html5_video())

def viz_trial(json_path, trial_number, figure_size = 18, pixel_density = 50):
    assert trial_number > 0, "Trial number starts from 1"
    json_data = load_trial_data(json_path)
    trial_data = json_data['trial_data']
    config = json_data['config']

    # Get the trial data for the specified trial number
    x_positions = trial_data[trial_number]
    all_frames = []
    for x_position in x_positions:
        frame = render_lr_frame(x_position, 1.0, config['LEFT_RIGHT_LENGTH'], pixel_density)
        all_frames.append(frame)
    all_frames = np.array(all_frames)
    return display_video(all_frames, framerate=config['FRAMES_PER_SECOND'], skip_t = 1, max_figsize_width = figure_size)


def create_negative_binomial_interactive_plot():
    """Create an interactive plot for exploring negative binomial distributions"""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_negative_binomial(rate, prob):
        """Plot samples from negative binomial distribution with given parameters"""
        num_samples = 1000
        
        # Convert parameters for numpy's negative binomial
        # numpy uses (n, p) where n is number of successes, p is probability of success
        # rate parameter corresponds to n, prob is p
        n = rate
        p = prob
        
        # Sample from negative binomial distribution using numpy and shift by 1
        samples = np.random.negative_binomial(n, p, num_samples) + 1
        
        # Clear previous output and create new plot
        clear_output(wait=True)
        
        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        
        # Create bins that are centered on integer values (starting from 1)
        max_val = int(np.max(samples))
        bins = np.arange(0.5, max_val + 1.5, 1)  # Bins centered on integers starting from 1
        
        plt.hist(samples, bins=bins, 
                 alpha=0.7, color='lightblue', edgecolor='black', density=True)
        plt.xlabel('Number of Frames Between Switches')
        plt.ylabel('Probability Density')
        plt.title(f'Shifted Negative Binomial Distribution (min=1)\nRate: {rate:.2f}, Probability (p): {prob:.3f}')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=-0.5)  # Set X axis lower limit to -0.5 to show 0 tick
        
        # Add statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {mean_val:.3f}')
        plt.legend()
        
        plt.tight_layout()
        return fig
        # plt.show()
        # plt.close()

    # Create interactive sliders
    rate_slider = widgets.FloatSlider(
        value=40.0,
        min=0.1,
        max=200.0,
        step=0.1,
        description='Rate:',
        style={'description_width': 'initial'}
    )

    prob_slider = widgets.FloatSlider(
        value=0.5,
        min=0.01,
        max=0.99,
        step=0.01,
        description='Probability (p):',
        style={'description_width': 'initial'}
    )

    # Create interactive plot
    interactive_plot = widgets.interactive(plot_negative_binomial, 
                                         rate=rate_slider, 
                                         prob=prob_slider)

    return interactive_plot

def create_geometric_interactive_plot():
    """Create an interactive plot for exploring geometric distributions"""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_geometric(prob):
        """Plot samples from geometric distribution with given parameters"""
        num_samples = 1000
        
        # Sample from shifted geometric distribution
        samples = sample_shifted_geometric(num_samples, prob)
        
        # Clear previous output and create new plot
        clear_output(wait=True)
        
        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        
        # Create bins that are centered on integer values (starting from 1)
        max_val = int(np.max(samples))
        bins = np.arange(0.5, max_val + 1.5, 1)  # Bins centered on integers starting from 1
        
        plt.hist(samples, bins=bins, 
                 alpha=0.7, color='lightblue', edgecolor='black', density=True)
        plt.xlabel('Number of Frames Between Switches')
        plt.ylabel('Probability Density')
        plt.title(f'Shifted Geometric Distribution (min=1)\nProbability (p): {prob:.3f}')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=-0.5)  # Set X axis lower limit to -0.5 to show 0 tick
        
        # Add statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {mean_val:.3f}')
        plt.legend()
        
        plt.tight_layout()
        return fig

    # Create interactive slider
    prob_slider = widgets.FloatSlider(
        value=0.5,
        min=0.01,
        max=0.99,
        step=0.01,
        description='Probability (p):',
        style={'description_width': 'initial'}
    )

    # Create interactive plot
    interactive_plot = widgets.interactive(plot_geometric, 
                                         prob=prob_slider)

    return interactive_plot

def create_truncated_normal_interactive():
    """Create interactive visualization for truncated discrete normal distribution"""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np
    def plot_truncated_normal(mean, std):
        """Plot samples from truncated discrete normal distribution with given parameters"""
        num_samples = 1000
        
        # Sample from truncated discrete normal distribution
        samples = sample_discrete_truncated_normal(num_samples, mean, std)
        
        # Clear previous output and create new plot
        clear_output(wait=True)
        
        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        
        # Create bins that are centered on integer values (starting from 1)
        max_val = int(np.max(samples))
        bins = np.arange(0.5, max_val + 1.5, 1)  # Bins centered on integers starting from 1
        
        plt.hist(samples, bins=bins, 
                 alpha=0.7, color='lightgreen', edgecolor='black', density=True)
        plt.xlabel('Number of Frames Between Switches')
        plt.ylabel('Probability Density')
        plt.title(f'Truncated Discrete Normal Distribution (min=1)\nMean: {mean:.2f}, Std: {std:.2f}')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=-0.5)  # Set X axis lower limit to -0.5 to show 0 tick
        
        # Add statistics
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        plt.axvline(sample_mean, color='red', linestyle='--', alpha=0.7, 
                    label=f'Sample Mean: {sample_mean:.3f}')
        plt.legend()
        
        plt.tight_layout()
        return fig

    # Create interactive sliders
    mean_slider = widgets.FloatSlider(
        value=150.0,
        min=1.0,
        max=500.0,
        step=0.1,
        description='Mean:',
        style={'description_width': 'initial'}
    )
    
    std_slider = widgets.FloatSlider(
        value=50.0,
        min=0.1,
        max=500.0,
        step=0.1,
        description='Std Dev:',
        style={'description_width': 'initial'}
    )

    # Create interactive plot
    interactive_plot = widgets.interactive(plot_truncated_normal, 
                                         mean=mean_slider,
                                         std=std_slider)

    return interactive_plot


def sample_shifted_negative_binomial(num_samples, rate, prob, seed=None):
    """Sample from a shifted negative binomial distribution (minimum value = 1)
    
    Args:
        num_samples: Number of samples to generate
        rate: Rate parameter (corresponds to n in numpy's negative_binomial)
        prob: Probability parameter (p in numpy's negative_binomial)
        seed: Random seed for reproducibility
    
    Returns:
        Array of samples from shifted negative binomial distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Sample from negative binomial distribution using numpy and shift by 1
    samples = np.random.negative_binomial(rate, prob, num_samples) + 1
    return samples

def sample_discrete_truncated_normal(num_samples, mean, std, seed=None):
    """Sample from a truncated normal distribution with discrete values
    
    Args:
        num_samples: Number of samples to generate
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        seed: Random seed for reproducibility
    
    Returns:
        Array of discrete samples from truncated normal distribution (minimum value = 1)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Sample from truncated normal distribution (lower bound = 1, no upper bound)
    from scipy.stats import truncnorm
    
    # Calculate the bounds in terms of standard deviations from the mean
    lower_bound = 1
    a = (lower_bound - mean) / std  # lower bound in standard deviations
    b = np.inf  # no upper bound
    
    # Sample from truncated normal
    continuous_samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=num_samples)
    
    # Convert to discrete values by rounding and ensuring minimum value is 1
    discrete_samples = np.maximum(1, np.round(continuous_samples).astype(int))
    
    return discrete_samples


def sample_shifted_geometric(num_samples, prob, seed=None):
    """Sample from a shifted geometric distribution (minimum value = 1)
    
    Args:
        num_samples: Number of samples to generate
        prob: Probability parameter (p in geometric distribution)
        seed: Random seed for reproducibility
    
    Returns:
        Array of samples from shifted geometric distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Sample from geometric distribution using numpy and shift by 1
    samples = np.random.geometric(prob, num_samples) + 1
    return samples


def generate_stimuli_and_trials(
    stimuli_name='lr_v1',
    initial_random_seed=42,
    frames_per_second=20,
    max_trial_seconds=10,
    min_trial_seconds=6,
    min_seconds_between_switches=None,
    max_num_switches=None,
    use_semi_markov_switching=True,
    semi_markov_distribution_type='discrete_normal',
    semi_markov_distribution_params={'rate': 75.0, 'prob': 0.3, 'mean': 100.0, 'std': 75.0},
    direction_flip_prob=0.025,
    speed=0.065,
    left_right_length=13,
    uniform_starting_position=True,
    uniform_starting_fractional_range_from_midpoint=1.0,
    uniform_starting_position_binning_length=0.5,
    non_uniform_starting_position=None,
    requested_num_trials=528,
    batch_size=1000,
    total_num_trials=100000
):
    """
    Generate stimuli and trials for the left-right task.
    
    Returns:
        dict: Contains all generated data including:
            - final_trial_data: Dictionary of trial position data
            - final_switch_data: Dictionary of switch information
            - all_positions_over_time: All position data
            - all_ending_indices: Trial ending indices
            - all_same_side_over_time: Same side outcomes
            - trial_indices_dict: Trial indices by bin
            - hypers: Hyperparameters used
            - config: Configuration parameters
    """
    import copy
    import numpy as np
    from tqdm import tqdm
    from genjax import ChoiceMapBuilder as C
    
    # Calculate the uniform starting range based on the fractional range parameter
    original_range_start = 0
    original_range_end = left_right_length - 1
    original_range_length = original_range_end - original_range_start
    
    # Calculate mid point of the original range
    range_midpoint = original_range_length / 2
    
    # Calculate the new uniform range based on the fractional parameter
    uniform_range_half_width = uniform_starting_fractional_range_from_midpoint * original_range_length / 2
    uniform_range_start = range_midpoint - uniform_range_half_width
    uniform_range_end = range_midpoint + uniform_range_half_width
    uniform_range_length = uniform_range_end - uniform_range_start
    
    # Calculate number of bins based on the actual uniform range, not the full track length
    num_uniform_bins = int(uniform_range_length / uniform_starting_position_binning_length)
    num_valid_trials_per_bin = int(requested_num_trials / num_uniform_bins)
    
    # Validation checks
    if requested_num_trials % num_uniform_bins != 0:
        raise ValueError(f"Requested number of trials {requested_num_trials} is not divisible by the number of bins {num_uniform_bins}")
    
    if num_valid_trials_per_bin % 2 != 0:
        raise ValueError(f"Number of valid trials per bin {num_valid_trials_per_bin} is not divisible by 2")
    
    if non_uniform_starting_position is not None and uniform_starting_position:
        raise ValueError("NON_UNIFORM_STARTING_POSITION and UNIFORM_STARTING_POSITION cannot both be used")
    
    if total_num_trials % batch_size != 0:
        raise ValueError(f"TOTAL_NUM_TRIALS ({total_num_trials}) must be divisible by BATCH_SIZE ({batch_size})")
    
    # Print configuration
    print(f"\033[94m🔄 Switching Behavior Configuration:\033[0m")
    if use_semi_markov_switching:
        print(f"   • Using \033[93mSemi-Markov Process\033[0m for direction switching")
        print(f"   • Distribution type: \033[93m{semi_markov_distribution_type}\033[0m")
        
        if semi_markov_distribution_type == 'geometric':
            relevant_params = {'prob': semi_markov_distribution_params['prob']}
        elif semi_markov_distribution_type == 'negative_binomial':
            relevant_params = {
                'rate': semi_markov_distribution_params['rate'],
                'prob': semi_markov_distribution_params['prob']
            }
        elif semi_markov_distribution_type == 'discrete_normal':
            relevant_params = {
                'mean': semi_markov_distribution_params['mean'],
                'std': semi_markov_distribution_params['std']
            }
        else:
            relevant_params = semi_markov_distribution_params
        
        print(f"   • Distribution parameters: \033[93m{relevant_params}\033[0m")
    else:
        print(f"   • Using \033[93mBernoulli Process\033[0m for direction switching")
        print(f"   • Direction flip probability: \033[93m{direction_flip_prob}\033[0m per frame")
    
    # Setup hyperparameters
    diameter = jnp.float32(1.0)
    hypers = LR_Hyperparams.create(
        max_speed=jnp.float32(diameter*speed*2),
        diameter=diameter,
        scene_length=jnp.float32(left_right_length * diameter),
        pixel_density=jnp.float32(50.0),
        pos_noise_std=jnp.float32(0.0),
        speed_noise_std=jnp.float32(0.0),
        direction_flip_prob=jnp.float32(direction_flip_prob),
        masked_occluders=jnp.array([[0,0]]), 
        left_right_labels=jnp.array([-1.0, 1.0]),
        is_semi_markov_switching=jnp.bool_(use_semi_markov_switching)
    )
    
    # Scale speed to diameter
    speed = float(speed * diameter)
    
    # Initialize random number generator
    key = jax.random.PRNGKey(initial_random_seed)
    
    # Get number of batches
    num_batches = (total_num_trials + batch_size - 1) // batch_size
    
    # Get total num frames
    total_num_frames = int(max_trial_seconds * frames_per_second)
    
    # Get mid point
    mid_point = (left_right_length / 2) - 0.5
    
    # Set starting position
    if non_uniform_starting_position is not None:
        starting_position = non_uniform_starting_position
    else:
        starting_position = mid_point
    
    # Generate switch times
    if use_semi_markov_switching:
        numpy_seed = initial_random_seed + 1
        switch_time_array = jnp.array(generate_switch_times(
            frames_per_second=frames_per_second, 
            max_trial_seconds=max_trial_seconds, 
            distribution_type=semi_markov_distribution_type,
            distribution_params=semi_markov_distribution_params,
            n_trials=total_num_trials, 
            seed=numpy_seed
        ))
    else:
        switch_time_array = jnp.zeros((total_num_trials, total_num_frames))
    
    batched_switch_time_array = jnp.array_split(switch_time_array, num_batches)
    
    # Initialize choice maps
    key, chm_key = jax.random.split(key)
    batched_chms = []
    if uniform_starting_position:
        for i in tqdm(range(num_batches), desc='setting choicemaps'):
            uniform_samples = jax.random.uniform(chm_key, (batch_size,))
            starting_positions = uniform_range_start + uniform_samples * uniform_range_length
            chm = C['x'].set(jnp.float32(starting_positions))
            chm = chm.at['speed'].set(jnp.full(batch_size, jnp.float32(speed)))
            batched_chms.append(chm)
    else:
        for i in tqdm(range(num_batches), desc='setting choicemaps'):
            chm = jax.vmap(lambda _: C['x'].set(jnp.float32(starting_position)))(jnp.arange(batch_size))
            chm = chm.at['speed'].set(jnp.full(batch_size, jnp.float32(speed)))
            batched_chms.append(chm)
    
    # Initialize states
    batched_init_states = []
    key, init_key = jax.random.split(key)
    batch_init_keys = jax.random.split(init_key, num_batches)
    for i in tqdm(range(num_batches), desc='initializing states'):
        all_batched_init_keys = jax.random.split(batch_init_keys[i], batch_size)
        batched_trs, _ = jax.vmap(lr_init_importance, in_axes=(0, 0, None))(all_batched_init_keys, batched_chms[i], (hypers,))
        batched_init_states.append(batched_trs.retval)
    
    # Simulation loop
    def scan_inner_loop(carry, i):
        key, states, switch_time_array_batch = carry
        key, batch_simulate_key = jax.random.split(key)
        all_batch_simulate_keys = jax.random.split(batch_simulate_key, batch_size)
        batched_states = jax.vmap(lr_stepper_simulate, in_axes=(0, (0, 0)))(all_batch_simulate_keys, (states, switch_time_array_batch)).retval
        return (key, batched_states, switch_time_array_batch), batched_states
    
    key, scan_key = jax.random.split(key)
    batch_scan_keys = jax.random.split(scan_key, num_batches)
    
    all_positions_over_time = []
    all_hit_boundary_over_time = []
    all_ending_indices = []
    all_same_side_over_time = []
    
    for i in tqdm(range(num_batches), desc='simulating states'):
        switch_time_array_batch = batched_switch_time_array[i]
        _, states_over_time = jax.lax.scan(scan_inner_loop, (batch_scan_keys[i], batched_init_states[i], switch_time_array_batch), jnp.arange(total_num_frames))
        all_positions_over_time.append(np.array(init_step_concat(batched_init_states[i].x, states_over_time.x)))
        all_hit_boundary_over_time.append(np.any(states_over_time.hit_boundary, axis=0))
        ending_indices = np.where(
            np.any(states_over_time.hit_boundary, axis=0), 
            np.argmax(states_over_time.hit_boundary, axis=0), 
            -1)
        all_ending_indices.append(ending_indices)
        
        if uniform_starting_position:
            started_left_side = batched_init_states[i].x <= mid_point
            ending_positions = states_over_time.x[ending_indices, jnp.arange(len(ending_indices))]
            ended_left_side = ending_positions <= mid_point
            all_same_side_over_time.append(np.array(ended_left_side) == np.array(started_left_side))
        else:
            if i % 2 == 0:
                all_same_side_over_time.append(np.array(np.zeros(batch_size)).astype(bool))
            else:
                all_same_side_over_time.append(np.array(np.ones(batch_size)).astype(bool))
    
    # Concatenate results
    all_positions_over_time = np.concatenate(all_positions_over_time, axis=1).T
    all_hit_boundary_over_time = np.concatenate(all_hit_boundary_over_time, axis=0)
    all_ending_indices = np.concatenate(all_ending_indices, axis=0)
    all_same_side_over_time = np.concatenate(all_same_side_over_time, axis=0)
    
    # Filter by timing criteria
    min_required_ending_index = frames_per_second * min_trial_seconds - 1
    all_hit_after_min_required_ending_index = all_ending_indices >= min_required_ending_index
    percentage_hit_after_min_required_ending_index = (np.sum(all_hit_after_min_required_ending_index) / len(all_hit_after_min_required_ending_index)) * 100
    
    total_trials = len(all_hit_boundary_over_time)
    trials_that_hit = np.sum(all_hit_boundary_over_time)
    percentage_hit_boundary = (trials_that_hit / total_trials) * 100
    
    print(f"\033[94m📊 Trial Statistics:\033[0m")
    print(f"   • Percentage of trials that ended within \033[93m{max_trial_seconds}\033[0m seconds: \033[92m{percentage_hit_boundary:.1f}%\033[0m")
    print(f"   • Percentage of trials that hit boundary after \033[93m{min_trial_seconds}\033[0m seconds: \033[92m{percentage_hit_after_min_required_ending_index:.1f}%\033[0m")
    
    valid_trials = all_hit_boundary_over_time & all_hit_after_min_required_ending_index
    
    total_trials = len(valid_trials)
    trials_that_hit = np.sum(valid_trials)
    percentage_hit = (trials_that_hit / total_trials) * 100
    
    print(f"\n\033[96m✅ Valid Trials (Timing Criteria):\033[0m")
    print(f"   • Generated \033[92m{trials_that_hit}\033[0m valid trials")
    print(f"   • Criteria: ended after \033[93m{min_trial_seconds}s\033[0m and hit boundary within \033[93m{max_trial_seconds}s\033[0m")
    print(f"   • Success rate: \033[92m{percentage_hit:.1f}%\033[0m of \033[93m{total_trials:,}\033[0m total trials")
    
    # Apply timing filter
    all_positions_over_time = all_positions_over_time[valid_trials]
    all_ending_indices_inclusive = copy.deepcopy(all_ending_indices)
    all_ending_indices = all_ending_indices[valid_trials]
    all_same_side_over_time = all_same_side_over_time[valid_trials]
    valid_trials = valid_trials[valid_trials]
    
    # Apply switch criteria filtering
    trials_before_switch_filter = len(all_positions_over_time)
    switch_valid_trials = np.ones(len(all_positions_over_time), dtype=bool)
    
    min_time_filtered = 0
    max_switches_filtered = 0
    
    # Check minimum time between switches
    if min_seconds_between_switches is not None:
        min_frames_between_switches = int(min_seconds_between_switches * frames_per_second)
        
        max_length = all_positions_over_time.shape[1]
        padded_positions = all_positions_over_time + diameter/2
        
        valid_mask = np.zeros_like(padded_positions, dtype=bool)
        for i, end_idx in enumerate(all_ending_indices):
            valid_mask[i, :end_idx + 2] = True
        
        diffs = np.diff(padded_positions, axis=1)
        signs = np.sign(diffs)
        direction_changes = np.diff(signs, axis=1)
        switch_masks = np.abs(direction_changes) > 0
        switch_masks = switch_masks & valid_mask[:, 2:]
        
        for i in range(len(all_positions_over_time)):
            switch_indices = np.where(switch_masks[i])[0] + 1
            
            if len(switch_indices) > 0:
                all_switch_points = np.concatenate([[0], switch_indices, [all_ending_indices[i] + 1]])
                time_between_switches = np.diff(all_switch_points)
                if np.any(time_between_switches < min_frames_between_switches):
                    switch_valid_trials[i] = False
                    min_time_filtered += 1
    
    # Check maximum number of switches
    if max_num_switches is not None:
        valid_indices = np.where(switch_valid_trials)[0]
        
        if len(valid_indices) > 0:
            valid_positions = all_positions_over_time[valid_indices] + diameter/2
            
            valid_mask = np.zeros_like(valid_positions, dtype=bool)
            for idx, i in enumerate(valid_indices):
                valid_mask[idx, :all_ending_indices[i] + 2] = True
            
            diffs = np.diff(valid_positions, axis=1)
            signs = np.sign(diffs)
            direction_changes = np.diff(signs, axis=1)
            switch_masks = np.abs(direction_changes) > 0
            switch_masks = switch_masks & valid_mask[:, 2:]
            
            num_switches = np.sum(switch_masks, axis=1)
            exceeds_max = num_switches > max_num_switches
            filtered_indices = valid_indices[exceeds_max]
            
            for i in filtered_indices:
                switch_valid_trials[i] = False
                max_switches_filtered += 1
    
    trials_after_switch_filter = np.sum(switch_valid_trials)
    total_switch_filtered = trials_before_switch_filter - trials_after_switch_filter
    switch_filter_percentage = (trials_after_switch_filter / trials_before_switch_filter) * 100
    
    if min_seconds_between_switches is None and max_num_switches is None:
        print(f"\n\033[96m🔄 Switch Criteria Filtering:\033[0m")
        print(f"   • Skipping switch filtering (both min_seconds_between_switches and max_num_switches are None)")
        print(f"   • Keeping all \033[93m{trials_before_switch_filter}\033[0m timing-valid trials")
    else:
        print(f"\n\033[96m🔄 Switch Criteria Filtering:\033[0m")
        print(f"   • Starting with \033[93m{trials_before_switch_filter}\033[0m timing-valid trials")
        
        if min_seconds_between_switches is not None:
            print(f"   • Minimum time between switches: \033[93m{min_seconds_between_switches}s\033[0m")
            print(f"     - Trials filtered out: \033[91m{min_time_filtered}\033[0m")
        
        if max_num_switches is not None:
            print(f"   • Maximum number of switches: \033[93m{max_num_switches}\033[0m")
            print(f"     - Trials filtered out: \033[91m{max_switches_filtered}\033[0m")
        
        print(f"   • \033[91mTotal trials filtered out:\033[0m \033[91m{total_switch_filtered}\033[0m")
        print(f"   • \033[92mTrials passing all switch criteria:\033[0m \033[92m{trials_after_switch_filter}\033[0m")
        print(f"   • Success rate: \033[92m{switch_filter_percentage:.1f}%\033[0m of timing-valid trials")
    # Apply switch filter
    all_positions_over_time = all_positions_over_time[switch_valid_trials]
    all_ending_indices = all_ending_indices[switch_valid_trials]
    all_same_side_over_time = all_same_side_over_time[switch_valid_trials]
    
    # Extract starting positions
    starting_positions = all_positions_over_time[:, 0]
    
    # Create trial indices dictionary
    trial_indices_dict = create_trial_indices_by_bin(
        num_uniform_bins, 
        starting_positions, 
        all_same_side_over_time, 
        uniform_range_start, 
        uniform_starting_position_binning_length
    )
    
    bin_sizes = np.array([[len(bin_dict['same_side']), len(bin_dict['different_side'])] for bin_dict in trial_indices_dict.values()])
    smallest_bin_size = np.min(bin_sizes)
    enough_trials = smallest_bin_size >= (num_valid_trials_per_bin // 2)
    
    if not enough_trials:
        print(f"\n\033[91m⚠️  Warning: Insufficient Trials for Counterbalancing\033[0m")
        print(f"   • Smallest bin has only \033[91m{smallest_bin_size}\033[0m trials")
        print(f"   • Required: \033[93m{num_valid_trials_per_bin // 2}\033[0m trials per condition per bin")
        if uniform_starting_position:
            print(f"   • \033[96mSuggestion:\033[0m Adjust trial generation parameters")
            print(f"   • Current uniform range: \033[93m{uniform_range_start:.2f} to {uniform_range_end:.2f}\033[0m (length: \033[93m{uniform_range_length:.2f}\033[0m)")
            print(f"   • Fractional range from midpoint: \033[93m{uniform_starting_fractional_range_from_midpoint}\033[0m")
        else:
            print(f"   • \033[96mSuggestion:\033[0m Enable uniform starting positions (set UNIFORM_STARTING_POSITION to True)")
    else:
        # Select trials for counterbalancing
        if uniform_starting_position:
            selected_trial_indices = []
            for bin_dict in trial_indices_dict.values():
                selected_trial_indices.extend(bin_dict['same_side'][:num_valid_trials_per_bin//2])
                selected_trial_indices.extend(bin_dict['different_side'][:num_valid_trials_per_bin//2])
            
            selected_trial_indices = np.array(selected_trial_indices)
        else:
            selected_trial_indices = np.arange(requested_num_trials)
        
        assert len(selected_trial_indices) == requested_num_trials, f"Expected {requested_num_trials} trials, got {len(selected_trial_indices)}"
        
        print(f"\n\033[92m🎉 Success! Dataset Generation Complete\033[0m")
        print(f"   • Generated \033[92m{requested_num_trials}\033[0m trials with perfect counterbalancing")
        print(f"   • Balanced for trials ending on same vs. different sides")
        print(f"   • All trials meet your specified parameters:")
        print(f"     - Trial length: \033[93m{min_trial_seconds}s - {max_trial_seconds}s\033[0m")
        print(f"     - Uniform starting positions: \033[93m{uniform_starting_position}\033[0m")
        if uniform_starting_position:
            print(f"     - Starting range: \033[93m{uniform_range_start:.2f} to {uniform_range_end:.2f}\033[0m (fractional range: \033[93m{uniform_starting_fractional_range_from_midpoint}\033[0m)")
        print(f"     - Binning length: \033[93m{uniform_starting_position_binning_length}\033[0m")
        if use_semi_markov_switching:
            print(f"     - Switching behavior: \033[93mSemi-Markov Process\033[0m")
            print(f"     - Distribution: \033[93m{semi_markov_distribution_type}\033[0m")
            
            if semi_markov_distribution_type == 'geometric':
                relevant_params = {'prob': semi_markov_distribution_params['prob']}
            elif semi_markov_distribution_type == 'negative_binomial':
                relevant_params = {
                    'rate': semi_markov_distribution_params['rate'],
                    'prob': semi_markov_distribution_params['prob']
                }
            elif semi_markov_distribution_type == 'discrete_normal':
                relevant_params = {
                    'mean': semi_markov_distribution_params['mean'],
                    'std': semi_markov_distribution_params['std']
                }
            else:
                relevant_params = semi_markov_distribution_params
            
            print(f"     - Distribution parameters: \033[93m{relevant_params}\033[0m")
        else:
            print(f"     - Switching behavior: \033[93mBernoulli Process\033[0m")
            print(f"     - Direction flip probability: \033[93m{direction_flip_prob}\033[0m per frame")
        if min_seconds_between_switches is not None:
            print(f"     - Min time between switches: \033[93m{min_seconds_between_switches}s\033[0m")
        if max_num_switches is not None:
            print(f"     - Max number of switches: \033[93m{max_num_switches}\033[0m")
        
        # Create final trial data
        final_positions_over_time = all_positions_over_time[selected_trial_indices]
        final_ending_indices = all_ending_indices[selected_trial_indices]
        final_trial_data = {i+1: final_positions_over_time[i, :final_ending_indices[i] + 2] + diameter/2 for i in range(requested_num_trials)}
        
        # Compute switch data
        final_switch_data = {}
        for i in range(1, requested_num_trials + 1):
            posdata = final_trial_data[i]
            diff = np.diff(posdata)
            direction_changes = np.diff(np.sign(diff))
            switch_mask = np.abs(direction_changes) > 0
            switch_indices = np.where(switch_mask)[0] + 1
            all_switch_points = np.concatenate([[0], switch_indices, [len(posdata) - 1]])
            time_between_switches = np.diff(all_switch_points)
            num_switches = len(switch_indices)
            
            final_switch_data[i] = {
                'switch_indices': switch_indices,
                'time_between_switches': time_between_switches,
                'num_switches': num_switches
            }
        
        print(f"\n\033[96m➡️  Next Step:\033[0m Move to the next cell to save trial data in JSON format")
        
        # Create configuration dictionary
        config = {
            'REQUESTED_NUM_TRIALS': requested_num_trials,
            'STIMULI_NAME': stimuli_name,
            'FRAMES_PER_SECOND': frames_per_second,
            'MAX_TRIAL_SECONDS': max_trial_seconds,
            'MIN_TRIAL_SECONDS': min_trial_seconds,
            'MIN_SECONDS_BETWEEN_SWITCHES': min_seconds_between_switches,
            'MAX_NUM_SWITCHES': max_num_switches,
            'USE_SEMI_MARKOV_SWITCHING': use_semi_markov_switching,
            'SEMI_MARKOV_DISTRIBUTION_TYPE': semi_markov_distribution_type,
            'SEMI_MARKOV_DISTRIBUTION_PARAMS': semi_markov_distribution_params,
            'DIRECTION_FLIP_PROB': direction_flip_prob,
            'SPEED': speed,
            'LEFT_RIGHT_LENGTH': left_right_length,
            'INITIAL_RANDOM_SEED': initial_random_seed,
            'UNIFORM_STARTING_POSITION': uniform_starting_position,
            'UNIFORM_STARTING_FRACTIONAL_RANGE_FROM_MIDPOINT': uniform_starting_fractional_range_from_midpoint,
            'UNIFORM_STARTING_POSITION_BINNING_LENGTH': uniform_starting_position_binning_length,
            'NON_UNIFORM_STARTING_POSITION': non_uniform_starting_position,
            'BATCH_SIZE': batch_size,
            'TOTAL_NUM_TRIALS': total_num_trials,
        }
        
        return {
            'final_trial_data': final_trial_data,
            'final_switch_data': final_switch_data,
            'all_positions_over_time': all_positions_over_time,
            'all_ending_indices': all_ending_indices,
            'all_ending_indices_inclusive': all_ending_indices_inclusive,
            'all_same_side_over_time': all_same_side_over_time,
            'trial_indices_dict': trial_indices_dict,
            'hypers': hypers,
            'config': config,
            'uniform_range_start': uniform_range_start,
            'uniform_range_end': uniform_range_end,
            'uniform_range_length': uniform_range_length,
            'num_uniform_bins': num_uniform_bins,
            'num_valid_trials_per_bin': num_valid_trials_per_bin,
            'enough_trials': enough_trials,
            'mid_point': mid_point,
            'diameter': diameter,
            'selected_trial_indices': selected_trial_indices
        }
    
    return None  # Return None if not enough trials


def plot_trial_ending_times_distribution(all_ending_indices_inclusive, frames_per_second, max_trial_seconds, min_trial_seconds):
    """
    Plot histogram of trial ending times distribution.
    
    Args:
        all_ending_indices_inclusive: Array of ending indices for all trials (including -1 for incomplete trials)
        frames_per_second: Number of frames per second
        max_trial_seconds: Maximum trial duration in seconds
        min_trial_seconds: Minimum trial duration in seconds
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    plt.figure(figsize=(10, 6))
    
    # Filter out trials that didn't finish (marked as -1)
    valid_ending_indices = all_ending_indices_inclusive[all_ending_indices_inclusive != -1]
    valid_ending_times = valid_ending_indices / frames_per_second
    
    # Count trials that ended before MAX_TRIAL_SECONDS (completed naturally)
    num_completed_trials = len(valid_ending_times)
    total_trials = len(all_ending_indices_inclusive)
    completion_percentage = (num_completed_trials / total_trials) * 100
    
    # Count trials that are valid (between MIN_TRIAL_SECONDS and MAX_TRIAL_SECONDS)
    num_valid_trials = len(valid_ending_times[valid_ending_times >= min_trial_seconds])
    
    # Create histogram data
    counts, bin_edges = np.histogram(valid_ending_times, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Color bins based on whether they are >= MIN_TRIAL_SECONDS
    colors = ['lightcoral' if center < min_trial_seconds else 'skyblue' for center in bin_centers]
    
    # Plot bars with different colors
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.7, color=colors, edgecolor='black')
    
    plt.xlabel('Ending Frame (Seconds)')
    plt.ylabel('Number of Trials')
    plt.title(f'Distribution of Trial Ending Times\n({num_completed_trials} trials ended before {max_trial_seconds} seconds, {completion_percentage:.1f}%)\n({num_valid_trials} valid trials between {min_trial_seconds}-{max_trial_seconds} seconds)', ha='center')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0, right=max_trial_seconds)
    
    # Add legend to explain colors
    legend_elements = [Patch(facecolor='lightcoral', label=f'< {min_trial_seconds} seconds'),
                       Patch(facecolor='skyblue', label=f'>= {min_trial_seconds} seconds')]
    plt.legend(handles=legend_elements)
    
    plt.show()


def plot_starting_position_vs_trial_outcome(trial_indices_dict, uniform_range_start, uniform_starting_position_binning_length, 
                                           num_uniform_bins, num_valid_trials_per_bin, enough_trials, 
                                           left_right_length, mid_point, diameter):
    """
    Plot starting position vs trial outcome (stacked bar chart).
    
    Args:
        trial_indices_dict: Dictionary mapping bin indices to trial indices
        uniform_range_start: Starting position of uniform range
        uniform_starting_position_binning_length: Length of each bin
        num_uniform_bins: Number of uniform bins
        num_valid_trials_per_bin: Number of valid trials per bin
        enough_trials: Boolean indicating if there are enough trials for counterbalancing
        left_right_length: Length of the track
        mid_point: Midpoint of the track
        diameter: Ball diameter
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    plt.figure(figsize=(12, 6))
    
    bin_centers = list(trial_indices_dict.keys())
    same_side_counts = [len(trial_indices_dict[center]["same_side"]) for center in bin_centers]
    different_side_counts = [len(trial_indices_dict[center]["different_side"]) for center in bin_centers]
    
    # Calculate actual bin positions based on uniform range parameters
    # Adjust positions to reflect center of ball instead of left edge
    ball_radius = diameter / 2
    actual_bin_positions = [uniform_range_start + i * uniform_starting_position_binning_length + uniform_starting_position_binning_length/2 + ball_radius
                           for i in range(num_uniform_bins)]
    
    width = uniform_starting_position_binning_length * 0.8
    
    bars1 = plt.bar(actual_bin_positions, same_side_counts, width=width, 
                    label='Ended on Same Side', color='lightblue', 
                    edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(actual_bin_positions, different_side_counts, width=width, 
                    bottom=same_side_counts, label='Ended on Different Side', 
                    color='lightcoral', edgecolor='black', linewidth=0.5)
    
    # Check if we have enough trials for counterbalancing and add selection overlay
    required_per_condition = num_valid_trials_per_bin // 2
    if enough_trials:
        # Add overlay boxes to show selected trials
        for i, pos in enumerate(actual_bin_positions):
            same_count = same_side_counts[i]
            diff_count = different_side_counts[i]
            
            # Overlay for same side trials (bottom segment)
            if same_count >= required_per_condition:
                plt.bar(pos, required_per_condition, width=width*0.95, 
                       color='none', edgecolor='green', linewidth=2, 
                       linestyle='--', alpha=0.8)
            
            # Overlay for different side trials (top segment)
            if diff_count >= required_per_condition:
                plt.bar(pos, required_per_condition, width=width*0.95, 
                       bottom=same_count, color='none', edgecolor='green', 
                       linewidth=2, linestyle='--', alpha=0.8)
    
    # Add vertical lines for track boundaries and midpoint (adjusted for ball center)
    plt.axvline(x=ball_radius, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Left Track Boundary (Ball Center)')
    plt.axvline(x=left_right_length-ball_radius, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Right Track Boundary (Ball Center)')
    plt.axvline(x=mid_point + ball_radius, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Track Midpoint (Ball Center)')
    
    # Add vertical lines for uniform range boundaries (adjusted for ball center)
    plt.axvline(x=uniform_range_start + ball_radius, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Uniform Range Start (Ball Center)')
    plt.axvline(x=uniform_range_start + num_uniform_bins * uniform_starting_position_binning_length + ball_radius, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Uniform Range End (Ball Center)')
    
    plt.xlabel('Starting Position (Ball Center)')
    plt.ylabel('Number of Trials')
    
    # Create title with counterbalancing information
    if enough_trials:
        total_selected_trials = len(bin_centers) * 2 * required_per_condition
        title = f'Starting Position vs. Trial Outcomes (Stacked)\nCounterbalanced: {required_per_condition} trials per condition per bin (green dashed boxes = selected trials)\n{len(bin_centers)} bins × 2 conditions × {required_per_condition} trials per condition = {total_selected_trials} total trials'
    else:
        title = f'Starting Position vs. Trial Outcomes (Stacked)\nInsufficient trials for counterbalancing (need {required_per_condition} per condition per bin)'
    
    plt.title(title)
    
    # Update legend to include selection overlay and boundary lines
    legend_elements = [bars1, bars2]
    if enough_trials:
        legend_elements.append(Patch(facecolor='none', edgecolor='green', 
                                    linestyle='--', linewidth=3, 
                                    label='Selected Trials'))
    
    # Add boundary line legend elements
    legend_elements.extend([
        Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Track Boundaries'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Track Midpoint'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Uniform Range Bounds')
    ])
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis limits to show full track range with some padding (adjusted for ball center)
    plt.xlim(ball_radius - 0.5, left_right_length - ball_radius + 0.5)
    
    # Set x-axis ticks to show every integer
    x_min = int(ball_radius - 0.5) + 1
    x_max = int(left_right_length - ball_radius + 0.5)
    plt.xticks(range(x_min, x_max + 1))
    
    # Add count labels in the middle of each bar segment
    for i, (pos, same_count, diff_count) in enumerate(zip(actual_bin_positions, same_side_counts, different_side_counts)):
        # Label for same side (bottom segment)
        if same_count > 0:
            plt.text(pos, same_count / 2, str(same_count), 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Label for different side (top segment)
        if diff_count > 0:
            plt.text(pos, same_count + diff_count / 2, str(diff_count), 
                    ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add total count labels on top of bars
    total_counts = np.array(same_side_counts) + np.array(different_side_counts)
    for i, (pos, total) in enumerate(zip(actual_bin_positions, total_counts)):
        if total > 0:
            plt.text(pos, total + max(total_counts) * 0.01, str(total), 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_switching_behavior_distribution(final_switch_data, requested_num_trials, frames_per_second):
    """
    Plot distribution of switching behavior (number of switches and time between switches).
    
    Args:
        final_switch_data: Dictionary containing switch information for each trial
        requested_num_trials: Number of requested trials
        frames_per_second: Number of frames per second
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'FINAL Distribution of Switching Behavior (after post-filtering)')
    
    plt.subplot(1, 2, 1)
    num_switches_per_trial = [final_switch_data[i]['num_switches'] for i in range(1, requested_num_trials + 1)]
    
    # Create bins centered on integer values
    max_switches = max(num_switches_per_trial)
    bins = np.arange(-0.5, max_switches + 1.5, 1)
    
    plt.hist(num_switches_per_trial, bins=bins, 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Switches per Trial')
    plt.ylabel('Number of Trials')
    plt.title(f'Distribution of Switches per Trial\n({len(num_switches_per_trial)} trials)')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=-0.5)
    
    # Plot histogram of time between switches
    plt.subplot(1, 2, 2)
    all_times_between_switches = []
    for i in range(1, requested_num_trials + 1):
        times_in_seconds = final_switch_data[i]['time_between_switches'] / frames_per_second
        all_times_between_switches.extend(times_in_seconds)
    
    plt.hist(all_times_between_switches, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Time Between Switches (seconds)')
    plt.ylabel('Number of Switch Intervals')
    plt.title(f'Distribution of Time Between Switches\n({len(all_times_between_switches)} switch intervals)')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=-0.1)
    
    plt.tight_layout()
    plt.show()


def generate_switch_times(frames_per_second, max_trial_seconds, distribution_type, distribution_params, n_trials=1, seed=None):
    """Generate switch times for multiple trials using shifted geometric, negative binomial, or discrete normal distribution
    
    Args:
        frames_per_second: Number of frames per second
        max_trial_seconds: Maximum duration of each trial in seconds
        distribution_type: Either 'geometric', 'negative_binomial', or 'discrete_normal'
        distribution_params: Dictionary of distribution parameters
        n_trials: Number of trials to generate
        seed: Random seed for reproducibility
    
    Returns:
        Array of shape (n_trials, max_frames) where 1 indicates a switch event
    """
    rate = distribution_params['rate']
    prob = distribution_params['prob']
    mean = distribution_params['mean']
    std = distribution_params['std']
    
    # Calculate maximum number of frames per trial
    max_frames = int(frames_per_second * max_trial_seconds)
    
    # Estimate how many samples we need per trial (conservative estimate)
    # Assume average inter-switch interval and add buffer
    # at least 5 switches per trial to be safe
    if distribution_type == 'geometric':
        expected_switches_per_trial = max(5, max_frames // (1 / prob))  # rough estimate for geometric
    elif distribution_type == 'negative_binomial':
        expected_switches_per_trial = max(5, max_frames // (rate / prob))  # rough estimate for negative binomial
    elif distribution_type == 'discrete_normal':
        expected_switches_per_trial = max(5, max_frames // mean)  # rough estimate for discrete normal
    else:
        raise ValueError("distribution_type must be either 'geometric', 'negative_binomial', or 'discrete_normal'")
    
    samples_per_trial = int(expected_switches_per_trial * 1)  # 1x buffer
    
    # Generate large batch of samples for all trials
    total_samples_needed = n_trials * samples_per_trial
    current_seed = seed if seed is not None else 0
    
    # Keep generating samples until we have enough for all trials
    if distribution_type == 'geometric':
        all_samples = sample_shifted_geometric(total_samples_needed, prob, current_seed)
    elif distribution_type == 'negative_binomial':
        all_samples = sample_shifted_negative_binomial(total_samples_needed, rate, prob, current_seed)
    else:  # discrete_normal
        all_samples = sample_discrete_truncated_normal(total_samples_needed, mean, std, current_seed)
    
    # Reshape samples into trials
    samples_per_trial_actual = len(all_samples) // n_trials
    trial_samples = all_samples[:n_trials * samples_per_trial_actual].reshape(n_trials, samples_per_trial_actual)
    
    # Generate more samples if any trial doesn't have enough
    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        # Calculate cumulative sums for all trials
        cumsum_trials = np.cumsum(trial_samples, axis=1)
        
        # Check if all trials have enough samples to reach max_frames
        last_cumsum = cumsum_trials[:, -1]
        insufficient_trials = last_cumsum < max_frames
        
        if not np.any(insufficient_trials):
            break
            
        # Generate more samples for insufficient trials
        current_seed += 1
        if distribution_type == 'geometric':
            additional_samples = sample_shifted_geometric(total_samples_needed, prob, current_seed)
        elif distribution_type == 'negative_binomial':
            additional_samples = sample_shifted_negative_binomial(total_samples_needed, rate, prob, current_seed)
        else:  # discrete_normal
            additional_samples = sample_discrete_truncated_normal(total_samples_needed, mean, std, current_seed)
        
        additional_per_trial = len(additional_samples) // n_trials
        additional_trial_samples = additional_samples[:n_trials * additional_per_trial].reshape(n_trials, additional_per_trial)
        
        # Concatenate new samples
        trial_samples = np.concatenate([trial_samples, additional_trial_samples], axis=1)
        attempt += 1
    
    # Calculate final cumulative sums
    cumsum_trials = np.cumsum(trial_samples, axis=1)
    
    # Initialize result array
    switch_arrays = np.zeros((n_trials, max_frames), dtype=int)
    
    # Vectorized assignment of switch events
    for trial_idx in range(n_trials):
        valid_switches = cumsum_trials[trial_idx][cumsum_trials[trial_idx] < max_frames]
        if len(valid_switches) > 0:
            switch_arrays[trial_idx, valid_switches] = 1
    
    return switch_arrays