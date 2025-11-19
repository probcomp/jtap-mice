import jax
import jax.numpy as jnp

from jtap.utils import ModelOutput

def red_green_sensor_read(mo: ModelOutput):
    
    x = mo.x
    y = mo.y
    diameter = mo.diameter

    red_x, red_y, red_size_x, red_size_y = mo.red_sensor
    green_x, green_y, green_size_x, green_size_y = mo.green_sensor
   
    in_red = jnp.logical_not(
            jnp.any(
                jnp.array([
                    jnp.less_equal(x+diameter, red_x),
                    jnp.greater_equal(y, red_y + red_size_y),
                    jnp.greater_equal(x, red_x + red_size_x),
                    jnp.less_equal(y+diameter, red_y),
                ])
            )
        )
    
    in_green = jnp.logical_not(
            jnp.any(
                jnp.array([
                    jnp.less_equal(x+diameter, green_x),
                    jnp.greater_equal(y, green_y + green_size_y),
                    jnp.greater_equal(x, green_x + green_size_x),
                    jnp.less_equal(y+diameter, green_y),
                ])
            )
        )

    return jnp.where(jnp.logical_or(in_green, in_red), jnp.where(in_red, jnp.int8(1), jnp.int8(2)), jnp.int8(0))

red_green_sensor_readouts = jax.vmap(red_green_sensor_read)