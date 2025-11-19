import jax
import jax.numpy as jnp

from jtap_mice.utils import ModelOutput

def left_right_sensor_read(mo: ModelOutput):
    x = mo.x
    y = mo.y
    diameter = mo.diameter

    # Check ball hit left or right side (boundary)
    left_contact = x <= 0
    right_contact = x >= (mo.scene_dim[0] - diameter)

    # if hit left, output 0; if hit right, output 1; else output 2
    sensor_val = jnp.where(left_contact, jnp.int8(0),
                  jnp.where(right_contact, jnp.int8(1), jnp.int8(2)))

    return sensor_val

left_right_sensor_readouts = jax.vmap(left_right_sensor_read)