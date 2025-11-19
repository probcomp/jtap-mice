from dataclasses import replace
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from genjax import PythonicPytree


class SuperPytree(PythonicPytree):
    """A base class that extends PythonicPytree from GenJAX with additional utility methods.

    This class provides enhanced functionality for working with JAX PyTrees, including
    advanced indexing, reshaping, and array manipulation operations. It's designed to
    work seamlessly with JAX code and operations
    """

    @staticmethod
    def _tree_getitem(
        pytree, idx: Union[int, slice, jnp.ndarray, Tuple]
    ) -> "SuperPytree":
        """Static method version of __getitem__ for functional use.

        Args:
            pytree: The SuperPytree instance (or compatible PyTree) to index.
            idx: Index or indices for array access.

        Returns:
            A new PyTree with the indexed arrays.
        """

        def safe_index(v, idx):
            if not isinstance(v, jnp.ndarray):  # Ignore non-JAX arrays
                return v

            # Handle scalar arrays (0-dimensional)
            if v.ndim == 0:
                return v  # Cannot index into scalar arrays

            if isinstance(idx, tuple):  # Multi-dimensional indexing
                if len(idx) > v.ndim:  # Too many indices
                    idx = idx[: v.ndim]  # Trim excess indices
            else:  # Single index case
                idx = (idx,)  # Convert to tuple for consistency

            # Efficient bounds checking using vectorized operations
            int_indices = [i for i in idx if isinstance(i, int)]
            relevant_shapes = v.shape[: len(int_indices)]

            if int_indices and relevant_shapes:
                int_indices_array = jnp.array(int_indices)
                shapes_array = jnp.array(relevant_shapes)

                # Check positive bounds and negative bounds in one go
                out_of_bounds = jnp.logical_or(
                    int_indices_array >= shapes_array, int_indices_array < -shapes_array
                )

                if jnp.any(out_of_bounds):
                    bad_idx = int(jnp.argmax(out_of_bounds))
                    raise IndexError(
                        f"Index {int_indices[bad_idx]} is out of bounds for axis {bad_idx} with size {relevant_shapes[bad_idx]}"
                    )

            return v[idx]  # Apply valid indexing

        return jtu.tree_map(lambda v: safe_index(v, idx), pytree)

    def __getitem__(self, idx: Union[int, slice, jnp.ndarray, Tuple]) -> "SuperPytree":
        """Supports single and multi-dimensional indexing of JAX arrays within the PyTree.

        This method enables advanced indexing operations on JAX arrays within the PyTree
        structure. It handles both single and multi-dimensional indexing, automatically
        trimming excess indices to match array dimensions.
        """
        return SuperPytree._tree_getitem(self, idx)

    @staticmethod
    def _tree_flatten(pytree) -> "SuperPytree":
        """Flattens all JAX arrays in the PyTree to 1D arrays.

        Returns:
            A new PyTree where all JAX arrays have been flattened to 1D arrays.
            Non-JAX array elements remain unchanged.
        """
        return jtu.tree_map(
            lambda x: x.flatten() if isinstance(x, jnp.ndarray) else x, pytree
        )

    def flatten(self) -> "SuperPytree":
        return SuperPytree._tree_flatten(self)

    @staticmethod
    def _tree_reshape(pytree, shape: Sequence[int]) -> "SuperPytree":
        """Reshapes all JAX arrays in the PyTree to the specified shape. Non-JAX array elements remain unchanged.

        Args:
            shape: Target shape for the arrays. Must be compatible with the total
                  number of elements in each array.

        Returns:
            A new PyTree with reshaped arrays.

        Raises:
            ValueError: If the target shape is incompatible with the array's total
                      number of elements.
        """

        def safe_reshape(x, shape):
            if jnp.prod(jnp.array(x.shape)) == jnp.prod(jnp.array(shape)):
                return x.reshape(shape)
            raise ValueError(f"Cannot reshape array of shape {x.shape} to {shape}")

        return jtu.tree_map(
            lambda x: safe_reshape(x, shape)
            if (isinstance(x, jnp.ndarray) and x.ndim > 0)
            else x,
            pytree,
        )

    def reshape(self, shape: Sequence[int]) -> "SuperPytree":
        return SuperPytree._tree_reshape(self, shape)

    @staticmethod
    def _tree_expand_dims(pytree, axis: int) -> "SuperPytree":
        """Expands the dimensions of all JAX arrays in the PyTree by 1 at the specified axis.

        Args:
            axis: The axis at which to insert the new dimension.

        Returns:
            A new PyTree with expanded arrays. Non-JAX array elements remain unchanged.

        Raises:
            IndexError: If the axis is not valid for at least one array in the PyTree.
        """

        def safe_expand_dims(x, axis):
            if not isinstance(x, jnp.ndarray):
                return x
            ndim = x.ndim
            # Acceptable axis range: [-ndim-1, ndim]
            if not (-ndim - 1 <= axis <= ndim):
                # any jnp scalar (including nan, inf, etc.) out of axis range is treated
                # as a static value (like how a python literal scalar is treated here)
                if ndim == 0:
                    return x
                else:
                    raise IndexError(
                        f"Cannot expand dims at axis {axis} for array with shape {x.shape} (ndim={ndim})"
                    )
            return jnp.expand_dims(x, axis)

        return jtu.tree_map(lambda x: safe_expand_dims(x, axis), pytree)

    def expand_dims(self, axis: int) -> "SuperPytree":
        return SuperPytree._tree_expand_dims(self, axis)

    @staticmethod
    def _tree_reshape_first_dim(pytree, shape: Sequence[int]) -> "SuperPytree":
        """Reshapes only the first dimension of all JAX arrays in the PyTree. Non-JAX array elements remain unchanged.

        This method preserves all dimensions after the first one, only modifying
        the first dimension according to the provided shape.

        Args:
            shape: Target shape for the first dimension. Must be compatible with
                  the total number of elements in the first dimension.

        Returns:
            A new PyTree with reshaped arrays.

        Raises:
            ValueError: If the target shape is incompatible with the array's
                      first dimension.
        """

        def safe_reshape(x, shape):
            if (
                isinstance(x, jnp.ndarray)
                and jnp.prod(jnp.array(x.shape)) % jnp.prod(jnp.array(shape)) == 0
            ):
                return x.reshape((*shape, *x.shape[1:]))
            raise ValueError(f"Cannot reshape first dimension of {x.shape} to {shape}")

        return jtu.tree_map(
            lambda x: safe_reshape(x, shape)
            if (isinstance(x, jnp.ndarray) and x.ndim > 0)
            else x,
            pytree,
        )

    def reshape_first_dim(self, shape: Sequence[int]) -> "SuperPytree":
        return SuperPytree._tree_reshape_first_dim(self, shape)

    @staticmethod
    def _tree_split(pytree, num_splits: int, axis: int = 0) -> Sequence["SuperPytree"]:
        """
        Splits all JAX arrays (with ndim > 0) in the PyTree into multiple parts along the specified axis.

        Only arrays with sufficient dimensions and a valid axis are split; others are left unchanged.
        The axis is checked for each array: it must satisfy -ndim <= axis < ndim for that array.
        Arrays that do not meet this requirement are not split and are left as-is in all returned PyTrees.

        If no arrays in the PyTree have a valid axis for splitting, a ValueError is raised.
        All arrays that are split must have the same size along the specified axis, or a ValueError is raised.

        Args:
            num_splits: Number of equal parts to split the arrays into.
            axis: The axis along which to split (default: 0). Must be valid for all arrays to be split.
                  For each array, axis must satisfy -ndim <= axis < ndim, where ndim is the number of dimensions of the array.

        Returns:
            A sequence of PyTrees, each containing one part of the split arrays.
            Arrays for which the axis is not valid are left unchanged in all returned PyTrees.

        Raises:
            ValueError: If no JAX arrays with a valid axis are found in the PyTree,
                        or if the arrays to be split do not have the same size along the specified axis.
        """

        def can_split(x):
            if isinstance(x, jnp.ndarray):
                ndim = x.ndim
                # Acceptable axis range: [-ndim, ndim-1]
                return (-ndim <= axis < ndim) and (ndim > 0)
            return False

        # Find arrays that can be split and check their sizes along the axis
        arrays = []

        def collect_arrays(x):
            if can_split(x):
                arrays.append(x)
            return None

        jtu.tree_map(collect_arrays, pytree)
        if not arrays:
            raise ValueError(
                "No JAX arrays with valid ndim for axis found in the PyTree to split."
            )

        # Check that all arrays to be split have the same size along the specified axis
        def get_axis_size(x):
            ndim = x.ndim
            ax = axis if axis >= 0 else ndim + axis
            if x.shape[ax] == 0:
                raise ValueError(
                    f"Cannot split array with shape {x.shape} along axis {axis} because it has no elements"
                )
            return x.shape[ax]

        try:
            sizes = [get_axis_size(x) for x in arrays]
        except Exception as e:
            raise ValueError(
                f"Axis {axis} is not valid for all arrays to be split: {e}"
            )

        if not all(size == sizes[0] for size in sizes):
            raise ValueError(
                f"All JAX arrays to be split must have the same size along axis {axis}, got sizes: {sizes}"
            )

        def maybe_split(x, i):
            if can_split(x):
                # Normalize axis for array_split
                ndim = x.ndim
                ax = axis if axis >= 0 else ndim + axis
                splits = jnp.array_split(x, num_splits, axis=ax)
                return splits[i]
            return x

        return [
            jtu.tree_map(lambda x: maybe_split(x, i), pytree) for i in range(num_splits)
        ]

    def split(self, num_splits: int, axis: int = 0) -> Sequence["SuperPytree"]:
        return SuperPytree._tree_split(self, num_splits, axis)

    @staticmethod
    def _tree_unstack(pytree, axis: int = 0) -> Sequence["SuperPytree"]:
        """
        Unstacks all JAX arrays (with ndim > 0) in the PyTree along the specified axis.

        For each array, the axis is checked: it must satisfy -ndim <= axis < ndim for that array.
        Arrays for which the axis is not valid are left unchanged in all returned PyTrees.

        If no arrays in the PyTree have a valid axis for unstacking, a ValueError is raised.
        All arrays that are unstacked must have the same size along the specified axis, or a ValueError is raised.

        Args:
            axis: The axis along which to unstack (default: 0). For each array, axis must satisfy
                  -ndim <= axis < ndim, where ndim is the number of dimensions of the array.

        Returns:
            A sequence of PyTrees, each containing one slice of the unstacked arrays.
            Arrays for which the axis is not valid are left unchanged in all returned PyTrees.

        Raises:
            ValueError: If no JAX arrays with a valid axis are found in the PyTree,
                        or if the arrays to be unstacked do not have the same size along the specified axis.
        """

        def can_unstack(x):
            if isinstance(x, jnp.ndarray):
                ndim = x.ndim
                return (-ndim <= axis < ndim) and (ndim > 0)
            return False

        # Find arrays that can be unstacked and check their sizes along the axis
        arrays = []

        def collect_arrays(x):
            if can_unstack(x):
                arrays.append(x)
            return None

        jtu.tree_map(collect_arrays, pytree)
        if not arrays:
            raise ValueError(
                "No JAX arrays with valid ndim for axis found in the PyTree to unstack."
            )

        # Check that all arrays to be unstacked have the same size along the specified axis
        def get_axis_size(x):
            ndim = x.ndim
            ax = axis if axis >= 0 else ndim + axis
            if x.shape[ax] == 0:
                raise ValueError(
                    f"Cannot unstack array with shape {x.shape} along axis {axis} because it has no elements"
                )
            return x.shape[ax]

        try:
            sizes = [get_axis_size(x) for x in arrays]
        except Exception as e:
            raise ValueError(
                f"Axis {axis} is not valid for all arrays to be unstacked: {e}"
            )

        if not all(size == sizes[0] for size in sizes):
            raise ValueError(
                f"All JAX arrays to be unstacked must have the same size along axis {axis}, got sizes: {sizes}"
            )

        num_slices = sizes[0]

        # Pre-unstack all arrays in one pass
        unstacked_arrays = {}
        array_id = 0

        def preprocess_and_unstack(x):
            nonlocal array_id
            if can_unstack(x):
                ndim = x.ndim
                ax = axis if axis >= 0 else ndim + axis
                # Unstack all slices at once using moveaxis + reshape
                moved = jnp.moveaxis(x, ax, 0)  # Move unstacking axis to front
                slices = [moved[i] for i in range(num_slices)]  # Extract all slices
                unstacked_arrays[array_id] = slices
                current_id = array_id
                array_id += 1
                return current_id  # Return array ID as placeholder
            return x

        # First pass: collect unstacked arrays and create template with IDs
        template = jtu.tree_map(preprocess_and_unstack, pytree)

        # Second pass: reconstruct trees using pre-unstacked arrays
        def reconstruct_slice(slice_idx):
            def get_slice_value(x):
                if isinstance(x, int) and x in unstacked_arrays:
                    return unstacked_arrays[x][slice_idx]
                return x

            return jtu.tree_map(get_slice_value, template)

        return [reconstruct_slice(i) for i in range(num_slices)]

    def unstack(self, axis: int = 0) -> Sequence["SuperPytree"]:
        return SuperPytree._tree_unstack(self, axis)

    @staticmethod
    def _tree_swapaxes(pytree, axis1: int, axis2: int) -> "SuperPytree":
        """
        Swaps two axes of all JAX arrays in the PyTree. Non-JAX array elements remain unchanged.

        For each array, axes are checked: both must satisfy -ndim <= axis < ndim for that array,
        and the array must have at least 2 dimensions. Arrays for which the axes are not valid
        are left unchanged.

        Args:
            axis1: First axis to swap.
            axis2: Second axis to swap.

        Returns:
            A new PyTree with swapped axes for all arrays where the axes are valid.
            Arrays for which the axes are not valid are left unchanged.
        """

        def can_swapaxes(x):
            if isinstance(x, jnp.ndarray):
                ndim = x.ndim
                return (
                    (ndim > 1) and (-ndim <= axis1 < ndim) and (-ndim <= axis2 < ndim)
                )
            return False

        def swapaxes_(x):
            if can_swapaxes(x):
                return jnp.swapaxes(x, axis1, axis2)
            return x

        return jax.tree_util.tree_map(swapaxes_, pytree)

    def swapaxes(self, axis1: int, axis2: int) -> "SuperPytree":
        return SuperPytree._tree_swapaxes(self, axis1, axis2)

    @staticmethod
    def _tree_replace(pytree, *args, do_replace_none=False, **kwargs) -> "SuperPytree":
        """Replaces fields in the PyTree with new values.

        This method supports both dictionary-based and keyword-based replacement.
        It can recursively replace nested PyTree structures.

        Args:
            *args: A single dictionary argument containing field replacements.
            do_replace_none: If True, None values will replace existing fields.
                           If False, None values are ignored (default: False).
            **kwargs: Keyword arguments specifying field replacements.
                     Cannot be used together with *args.

        Returns:
            A new PyTree with the specified fields replaced.

        Raises:
            AssertionError: If both positional and keyword arguments are provided.
        """

        if len(kwargs) > 0:
            assert len(args) == 0, (
                "Cannot mix positional and keyword arguments in replace."
            )
            return SuperPytree._tree_replace(
                pytree, kwargs, do_replace_none=do_replace_none
            )

        assert len(args) == 1 and isinstance(args[0], dict), (
            "Expected a single dictionary argument."
        )
        new_fields = args[0]

        def recurse(k, v):
            current_val = getattr(pytree, k)
            if isinstance(v, dict) and isinstance(current_val, SuperPytree):
                return SuperPytree._tree_replace(
                    current_val, v, do_replace_none=do_replace_none
                )
            else:
                return v

        new_fields = {
            k: recurse(k, v)
            for k, v in new_fields.items()
            if do_replace_none or v is not None
        }
        return replace(pytree, **new_fields)  # type: ignore

    def replace(self, *args, do_replace_none=False, **kwargs) -> "SuperPytree":
        return SuperPytree._tree_replace(
            self, *args, do_replace_none=do_replace_none, **kwargs
        )