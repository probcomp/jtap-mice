import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


d2r = jax.jit(lambda x: jnp.float32(x) * jnp.pi / 180)
r2d = jax.jit(lambda x: jnp.float32(x) * 180 / jnp.pi)

def angle_diff_radians(a, b):
    return jnp.mod(a - b + jnp.pi, 2*jnp.pi) - jnp.pi

def softmax(logw):
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    return w / jnp.sum(w)

def angle_diff_deg(a, b):
    return r2d(angle_diff_radians(d2r(a), d2r(b)))

normalize_log_weights = lambda log_weights : log_weights - logsumexp(log_weights)
effective_sample_size = lambda log_weights : jnp.exp(-logsumexp(2. * normalize_log_weights(log_weights)))

check_invalid = lambda x: jnp.logical_or(jnp.isnan(x), jnp.isinf(x))
check_valid = lambda x: jnp.logical_not(check_invalid(x))