import jax
import jax.numpy as jnp
import jax.tree_util as jtu

def concat_pytrees(pytree_list, axis = 0):
    return jtu.tree_map(lambda *xs: jnp.concatenate(xs, axis = axis), *pytree_list)

def stack_pytrees(pytree_list, axis = 0):
    return jtu.tree_map(lambda *ys: jnp.stack(ys, axis = axis), *pytree_list)

# JAX pytree leaves only
def reshape_first_dim_pytree(pytree, shape):
    def reshape(x):
        new_shape = (*shape,*x.shape[1:])
        return x.reshape(new_shape)
    return jtu.tree_map(reshape, pytree)

def split_pytree(pytree, num_splits):
    return [jtu.tree_map(lambda x: jnp.array_split(x, num_splits)[i], pytree) for i in range(num_splits)]

def flatten_pytree(pytree):
    return jtu.tree_map(lambda x: x.flatten(), pytree)

def swap_axes_pytree(pytree, axis1, axis2):
    def swap_axes(x):
        if hasattr(x, 'shape') and len(x.shape) > max(axis1, axis2):
            return jnp.swapaxes(x, axis1, axis2)
        return x  # Leave non-array or insufficiently dimensional objects unchanged
    
    return jax.tree_util.tree_map(swap_axes, pytree)

def slice_pytree(pytree, i):
    return jtu.tree_map(lambda v : v[i], pytree)

slice_pt = slice_pytree

def safe_slice_pytree(pytree, i, unsafe_dim_len = jnp.inf):
    return jtu.tree_map(
        lambda v: v[i] if hasattr(v, 'shape') and 
        len(v.shape) > 0 and v.shape[0] > i 
        and v.shape[0] != unsafe_dim_len
        else v, 
        pytree
    )

def safe_fromend_slice_pytree(pytree, endoffset, unsafe_dim_len = jnp.inf):
    return jtu.tree_map(
        lambda v: v[:-endoffset] if hasattr(v, 'shape') and 
        len(v.shape) > 0 and v.shape[0] > 0 
        and v.shape[0] != unsafe_dim_len
        else v, 
        pytree
    )

def init_step_concat(init, steps):
    return jtu.tree_map(
        lambda a, b: jnp.concatenate([a[None, ...], b], axis=0), 
        init, steps
    )

def multislice_pytree(pytree, indices):
    return jax.vmap(slice_pytree, in_axes = (None, 0))(pytree, indices)


def flattened_multivmap(f, unflatten_output = True):
    """
    This flattened VMAP does a multi-vmap 
    over a flattened collapsed representation
    along with memory-saving array splits
    """
    def _get_vmapped(f, n):
        f = jax.jit(jax.vmap(f, in_axes = tuple(0 for _ in range(n))))
        return f

    def inner(map_encoding, *args):
        # NOTE THAT LAST DIMENSION MUST BE MULTIMAPPED FOR NOW
        assert map_encoding[-1] == 1
        # in map_encoding, 1 means multivmap and 0 means single vmap over flattened dimension
        n = len(args)
        assert len(map_encoding) == n
        vmapped_fn = _get_vmapped(f, n)

        multivmap_args = tuple(x for i,x in enumerate(args) if map_encoding[i] == 1)
        singlevmap_args = tuple(x for i,x in enumerate(args) if map_encoding[i] == 0)

        shapes = tuple([x.shape[0] for x in multivmap_args])
        # print(shapes)
        extended_args = jnp.meshgrid(*multivmap_args, indexing = 'ij')
        extended_args_flattened = []
        multi_count = 0
        for i in range(n):
            if map_encoding[i] == 1:
                extended_args_flattened.append(extended_args[multi_count].flatten())
                multi_count += 1
            else:
                extended_args_flattened.append(singlevmap_args[i - multi_count])
        extended_args_flattened = tuple(extended_args_flattened)
        # print([x.shape for x in extended_args_flattened])
        # extended_args_flattened = tuple(x.flatten() for x in extended_args)
        len_arg = extended_args_flattened[-1].shape[0]
        num_segs = max(len_arg//1000, 1)
        # print(len_arg, num_segs)
        extended_args_flattened_split = tuple(split_pytree(x,num_segs) for x in extended_args_flattened)
        split_outputs = []
        for i in range(num_segs):
            # print(f"seg {i+1} of {num_segs}")
            split_outputs.append(vmapped_fn(*tuple(x[i] for x in extended_args_flattened_split)))
        
        combined_output = concat_pytrees(split_outputs)
        
        if unflatten_output:
            return reshape_first_dim_pytree(combined_output, shapes)
        else:
            return combined_output
    
    return inner

def multivmap(f, num_vmap_dims = None):
    """
    Given a function `f` of `n` arguments, return a function
    on `n` jax vectors of arguments `a1, .., an`, which outputs an n-dimensional
    array `A` such that `A[i1, ..., in] = f(a1[i1], ..., an[in])`.
    """
    def _get_multivmapped(f, n, num_vmap_dims):
        if num_vmap_dims is None:
            num_vmap_dims = n
        for i in range(num_vmap_dims - 1, -1, -1):
            f = jax.vmap(f, in_axes=tuple(0 if j == i else None for j in range(n)))
        return f

    def inner(*args):
        n = len(args)
        return _get_multivmapped(f, n, num_vmap_dims)(*args)
    
    return inner