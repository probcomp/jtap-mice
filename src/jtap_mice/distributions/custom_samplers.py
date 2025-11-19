import numpy as np
from scipy.stats import truncnorm
from genjax import Pytree, ExactDensity
import jax
import jax.numpy as jnp

def truncated_normal_sample(key_seed, params, size):
    """
    Sample from a truncated normal distribution between lower and upper bounds.
    """
    if params is None:
        return None
    mean, std, lower, upper = params

    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=key_seed)

def discrete_normal_sample(key_seed, params, size):
    """
    Sample from a discrete approximation of a normal distribution.
    This is done by sampling from `discrete_values` with probabilities derived from the normal distribution.
    The random seed is fixed using key_seed.
    """
    if params is None:
        return None
    mean, std, discrete_values = params
    # Compute probabilities for each discrete value
    probabilities = np.exp(-0.5 * ((discrete_values - mean) ** 2) / (std ** 2))
    probabilities /= probabilities.sum()  # Normalize to sum to 1

    rng = np.random.RandomState(key_seed)
    return rng.choice(discrete_values, size=size, p=probabilities)

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