

import jax
import jax.numpy as jnp
from genjax import Pytree, ExactDensity
from jtap_mice.utils import slice_pt

@Pytree.dataclass
class PixelFlipLikelihood(ExactDensity):
    # HARDCODED TO HAVE 3 POSSIBLE PIXEL VALUES (0, 1, 2)
    def sample(self, key, render_args, pixel_corruption_prob, *args, **kwargs):
        rendered_image = render_scene(*render_args)
        
        # Apply pixel flip noise
        flip_key, noise_key = jax.random.split(key)
        
        # Determine which pixels to flip based on pixel_corruption_prob
        flip_mask = jax.random.uniform(flip_key, rendered_image.shape) < pixel_corruption_prob        
        
        # For each pixel, create a mask of valid alternatives (all values except the original)
        def sample_alternative_pixel(original_val, rng_key):
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

    def logpdf(self, obs_image, render_args, pixel_corruption_prob, *args, **kwargs):
        return jnp.sum(
                jnp.where(obs_image == render_scene(*render_args), jnp.log(1 - pixel_corruption_prob), jnp.log(pixel_corruption_prob))
        )
    
pixel_flip_likelihood = PixelFlipLikelihood()


def log_gaussian_tile(size: int, sigma: float) -> jnp.ndarray:
    """
    Create a log-probability Gaussian tile for spatial blurring.
    
    Args:
        size: Size of the tile (should be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Log-probability tile normalized to sum to 1 in probability space
    """
    sigma = jnp.where(jnp.equal(sigma, 0.0), 1e-20, sigma)
    one_directional_size = size // 2
    ax = jnp.arange(-one_directional_size, one_directional_size + 1.0)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = -(xx**2 + yy**2) / (2.0 * sigma**2)
    kernel = kernel - jax.nn.logsumexp(kernel)  # normalize
    return kernel

def get_latent_window(padded_latent, ij, tile_size):
    """Extract a window from the padded latent image at position ij."""
    return jax.lax.dynamic_slice(
        padded_latent,
        (ij[0], ij[1]),
        (2 * tile_size + 1, 2 * tile_size + 1),
    )

def compute_tile_log_prob(latent_window, pixel_value, log_tile):
    """Compute log probability for a pixel value using the Gaussian tile."""
    match_mask = (latent_window == pixel_value).astype(jnp.float32)
    return jax.nn.logsumexp(jnp.where(match_mask, log_tile, -jnp.inf))

def compute_mixed_log_prob(tile_log_prob, pixel_corruption_prob):
    """Mix tile and outlier probabilities using pixel_corruption_prob."""
    outlier_log_prob = jnp.log(1.0 / 3.0)
    log_probs_to_mix = jnp.array([
        jnp.log(1.0 - pixel_corruption_prob) + tile_log_prob,
        jnp.log(pixel_corruption_prob) + outlier_log_prob
    ])
    return jax.nn.logsumexp(log_probs_to_mix)

def create_pixel_indices(pix_y_shape, pix_x_shape):
    """Create flattened indices for all pixels in the image."""
    y_idxs, x_idxs = jnp.arange(pix_y_shape), jnp.arange(pix_x_shape)
    yy, xx = jnp.meshgrid(y_idxs, x_idxs, indexing='ij')
    return jnp.stack([yy.flatten(), xx.flatten()], axis=1)

def setup_tile_computation(render_args, tile_size_arr, σ_pixel_spatial):
    """Common setup for tile-based computations."""
    tile_size = tile_size_arr.shape[0]
    pix_x, pix_y = render_args[:2]
    
    # Render and pad the scene
    rendered_scene = render_scene(*render_args)
    padded_latent = jnp.pad(rendered_scene, ((tile_size, tile_size), (tile_size, tile_size)), mode='edge')
    
    # Create indices and tile
    indices = create_pixel_indices(pix_y.shape[0], pix_x.shape[0])
    log_tile = log_gaussian_tile((2*tile_size)+1, σ_pixel_spatial)
    
    return rendered_scene, padded_latent, indices, log_tile, tile_size

# @jax.jit
def logpdf_per_pixel(
    ij,
    obs_image,
    padded_latent,
    log_tile,
    tile_size_arr,
    pixel_corruption_prob
):
    """
    Compute log probability for a single pixel given the latent scene and observation model.
    
    The model combines:
    1. Gaussian tile sampling from matching pixels in the latent window
    2. Uniform outlier sampling with probability pixel_corruption_prob
    """
    tile_size = tile_size_arr.shape[0]
    latent_window = get_latent_window(padded_latent, ij, tile_size)
    obs_pixel_val = obs_image[ij[0], ij[1]]
    
    tile_log_prob = compute_tile_log_prob(latent_window, obs_pixel_val, log_tile)
    return compute_mixed_log_prob(tile_log_prob, pixel_corruption_prob)

logpdf_per_pixel_vmap = jax.vmap(logpdf_per_pixel, in_axes=(0, None, None, None, None, None))

@jax.jit
def sample_per_pixel(
    ij,
    key,
    padded_latent,
    log_tile,
    tile_size_arr,
    pixel_corruption_prob,
    image_power_beta = 1.0
):
    """
    Sample a single pixel value given the latent scene and observation model.
    
    The model combines:
    1. Gaussian tile sampling from matching pixels in the latent window
    2. Uniform outlier sampling with probability pixel_corruption_prob
    
    Temperature parameter image_power_beta tempers the entire likelihood for sampling.
    """
    tile_size = tile_size_arr.shape[0]
    latent_window = get_latent_window(padded_latent, ij, tile_size)
    
    # Create probability distribution over possible pixel values (0-2)
    pixel_values = jnp.array([0, 1, 2], dtype=jnp.int8)
    
    # Compute log probabilities for each possible pixel value
    def compute_log_prob_for_pixel_val(pix_val):
        tile_log_prob = compute_tile_log_prob(latent_window, pix_val, log_tile)
        # mixed_log_prob = compute_mixed_log_prob(tile_log_prob, pixel_corruption_prob)
        new_flip_prob = jnp.where(pix_val == 2, pixel_corruption_prob, 0.1*pixel_corruption_prob)
        mixed_log_prob = compute_mixed_log_prob(tile_log_prob, new_flip_prob)
        return image_power_beta * mixed_log_prob  # Apply temperature to entire likelihood
    
    log_probs = jnp.array([
        compute_log_prob_for_pixel_val(pix_val)
        for pix_val in pixel_values
    ])

    # Sample using categorical distribution with log probabilities
    return jax.random.categorical(key, logits=log_probs).astype(jnp.int8)

sample_per_pixel_vmap = jax.vmap(sample_per_pixel, in_axes=(0, 0, None, None, None, None, None))

@Pytree.dataclass
class GaussianTileLikelihood(ExactDensity):
    """
    Likelihood model that combines Gaussian tile blurring with uniform outlier noise.
    
    For each pixel, the model:
    1. With probability (1-pixel_corruption_prob): samples from a Gaussian tile centered on matching 
       pixels in the rendered latent scene
    2. With probability pixel_corruption_prob: samples uniformly from all 3 possible pixel values
    
    Temperature parameter image_power_beta tempers the entire likelihood.
    
    This ensures all pixel values have non-zero probability even if not present in the 
    rendered scene.
    """
    
    def sample(self, key, render_args, pixel_corruption_prob, tile_size_arr, σ_pixel_spatial, image_power_beta = 1.0, **kwargs):
        """
        Sample an observed image given render arguments and noise parameters.
        """
        rendered_scene, padded_latent, indices, log_tile, tile_size = setup_tile_computation(
            render_args, tile_size_arr, σ_pixel_spatial
        )
        
        # Generate keys for each pixel
        pixel_keys = jax.random.split(key, indices.shape[0])
        
        # Sample each pixel
        sampled_pixels = sample_per_pixel_vmap(
            indices,
            pixel_keys,
            padded_latent,
            log_tile,
            tile_size_arr,
            pixel_corruption_prob,
            image_power_beta
        )
        
        # Reshape back to image shape
        return sampled_pixels.reshape(rendered_scene.shape)

    def logpdf(self, obs_image, render_args, pixel_corruption_prob, tile_size_arr, σ_pixel_spatial, image_power_beta = 1.0, **kwargs):
        """
        Compute log probability of observed image given render arguments and noise parameters.
        """
        rendered_scene, padded_latent, indices, log_tile, tile_size = setup_tile_computation(
            render_args, tile_size_arr, σ_pixel_spatial
        )
        
        # Compute log probabilities for each pixel
        pixel_logpdfs = logpdf_per_pixel_vmap(
            indices,
            obs_image,
            padded_latent,
            log_tile,
            tile_size_arr,
            pixel_corruption_prob
        )

        # Apply temperature to the entire likelihood
        return image_power_beta * pixel_logpdfs.sum()


gaussian_tile_likelihood = GaussianTileLikelihood()

@jax.jit
def render_scene(pix_x, pix_y, diameter, x, y, masked_occluders, image_discretization):
    # Precompute the grid - note that pix_x corresponds to columns (width) and pix_y to rows (height)
    # Following image convention: first index is rows (y/height), second index is columns (x/width)
    y_vals, x_vals = jnp.meshgrid(pix_y, pix_x, indexing='ij')
    max_occluders = masked_occluders.flag.shape[0]

    # Initialize the image with shape (height, width) = (pix_y.shape[0], pix_x.shape[0])
    image = jnp.zeros((pix_y.shape[0], pix_x.shape[0]), dtype=jnp.int8)

    # Render the target with center snapped to closest pixel coordinates
    r = diameter / 2
    # Calculate center coordinates by adding radius to position
    target_center_x_spatial = x + r
    target_center_y_spatial = y + r
    
    # Snap center to closest values in pix_x and pix_y arrays
    target_center_x = pix_x[jnp.argmin(jnp.abs(pix_x - target_center_x_spatial))]
    target_center_y = pix_y[jnp.argmin(jnp.abs(pix_y - target_center_y_spatial))]
    
    # Use a small epsilon to handle floating point precision issues
    distance_squared = (x_vals - target_center_x + 0.5*image_discretization)**2 + (y_vals - target_center_y + 0.5*image_discretization)**2
    r_squared = r**2
    
    image = jnp.where((distance_squared) < (r_squared), jnp.int8(2), image)

    # Render occluders as pixel value 1
    for i in range(max_occluders):
        occluder_x, occluder_y, occluder_size_x, occluder_size_y = slice_pt(masked_occluders.value,i)
        image = jnp.where(masked_occluders.flag[i] & (x_vals >= occluder_x) & (y_vals >= occluder_y) & (x_vals < occluder_x + occluder_size_x) & (y_vals < occluder_y + occluder_size_y), jnp.int8(1), image)

    return image


# NOTE: SETTING LIKELIHOOD: Set the likelihood model to use. Never change this
# likelihood_model = gaussian_tile_likelihood
likelihood_model = pixel_flip_likelihood