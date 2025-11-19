from typing import Union, TypeVar, Any
import numpy as np
import jax.numpy as jnp
from attrs import define, field, validators
import functools

T = TypeVar('T', bound=Union[np.ndarray, jnp.ndarray, list, tuple, int, float, bool])

@define(frozen=True)
class StaticJnp:
    """A wrapper class for static JAX arrays.

    This class provides a way to store JAX arrays in a static context, with
    cached conversion to NumPy arrays for efficient comparison and hashing.

    Attributes:
        v: The JAX array being wrapped
    """

    v: jnp.ndarray = field(
        converter=jnp.array, validator=validators.instance_of(jnp.ndarray)
    )

    @functools.cached_property
    def np_v(self) -> np.ndarray:
        """Get the NumPy array representation of the JAX array.

        This property is cached for efficiency, as the conversion only needs
        to be performed once.

        Returns:
            np.ndarray: The NumPy array representation of the JAX array
        """
        return np.array(self.v)

    def __eq__(self, other: Any) -> bool:
        """Compare two StaticJnp instances for equality.

        Two instances are considered equal if their underlying NumPy arrays
        are equal.

        Args:
            other: Another StaticJnp instance to compare with

        Returns:
            bool: True if the arrays are equal, False otherwise
        """
        if not isinstance(other, StaticJnp):
            return False
        return np.array_equal(self.np_v, other.np_v)

    def __hash__(self) -> int:
        """Compute a hash value for the static JAX array.

        The hash is based on the binary representation of the NumPy array.
        This method is cached for efficiency.

        Returns:
            int: A hash value for the array
        """
        return hash(self.np_v.tobytes())

def snp(x: T) -> StaticJnp:
    """Create a StaticJnp wrapper around a JAX array.

    Args:
        x: Input data to convert to a JAX array and wrap in StaticJnp

    Returns:
        StaticJnp: A StaticJnp instance containing the JAX array
    """
    return StaticJnp(jnp.array(x))

def i_(x: T) -> jnp.ndarray:
    return jnp.int32(x)

def i8_(x: T) -> jnp.ndarray:
    return jnp.int8(x)

def f_(x: T) -> jnp.ndarray:
    return jnp.float32(x)

def b_(x: T) -> jnp.ndarray:
    return jnp.bool_(x)

def a_(x: T) -> jnp.ndarray:
    return jnp.array(x)