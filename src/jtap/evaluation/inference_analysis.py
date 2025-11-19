import jax
import jax.numpy as jnp

from jtap.utils import softmax

def compute_weight_component_correlations(jtap_data, run_idx=None):
    """
    Compute correlations between prev/incr/prop/grid weights and final weights
    across all timesteps in a batched manner.
    
    Args:
        jtap_data: JTAP data object
        run_idx: Index of the run to analyze (required if jtap_data has multiple runs)
    
    Returns:
        dict: Dictionary with keys 'prev', 'incr', 'prop', 'grid' containing
              arrays of correlations for each timestep
    """
    # Handle multiple runs
    if jtap_data.num_jtap_runs > 1:
        if run_idx is None:
            raise ValueError(f"Multiple JTAP runs detected ({jtap_data.num_jtap_runs} runs). Please provide run_idx.")
        if run_idx >= jtap_data.num_jtap_runs:
            raise ValueError(f"run_idx {run_idx} is out of range. Available runs: 0 to {jtap_data.num_jtap_runs - 1}")
        
        # Extract weight data for specific run
        weight_data_prev = jtap_data.inference.weight_data.prev_weights[run_idx]
        weight_data_incr = jtap_data.inference.weight_data.incremental_weights[run_idx]
        weight_data_prop = jtap_data.inference.weight_data.prop_weights[run_idx]
        weight_data_grid = jtap_data.inference.weight_data.grid_weights[run_idx]
        weight_data_final = jtap_data.inference.weight_data.final_weights[run_idx]
        max_steps = jtap_data.params.max_inference_steps[run_idx]  # Integer: number of inference timesteps
    else:
        # Single run case
        weight_data_prev = jtap_data.inference.weight_data.prev_weights
        weight_data_incr = jtap_data.inference.weight_data.incremental_weights
        weight_data_prop = jtap_data.inference.weight_data.prop_weights
        weight_data_grid = jtap_data.inference.weight_data.grid_weights
        weight_data_final = jtap_data.inference.weight_data.final_weights
        max_steps = jtap_data.params.max_inference_steps  # Integer: number of inference timesteps

    
    
    # Get all weights in batched form [timesteps, particles]
    w_prev_all = jnp.array([softmax(weight_data_prev[i]) for i in range(max_steps)])  # [max_steps, particles]
    w_incr_all = jnp.array([softmax(weight_data_incr[i]) for i in range(max_steps)])  # [max_steps, particles]
    w_prop_all = jnp.array([softmax(-weight_data_prop[i]) for i in range(max_steps)])  # [max_steps, particles]
    w_grid_all = jnp.array([softmax(-weight_data_grid[i]) for i in range(max_steps)])  # [max_steps, particles]
    w_final_all = jnp.array([softmax(weight_data_final[i]) for i in range(max_steps)])  # [max_steps, particles]
    
    def batch_correlate(a_batch, b_batch):
        """Compute correlation for each pair in the batch"""
        # Center the data
        a_centered = a_batch - jnp.mean(a_batch, axis=1, keepdims=True)
        b_centered = b_batch - jnp.mean(b_batch, axis=1, keepdims=True)
        
        # Compute dot products
        numerator = jnp.sum(a_centered * b_centered, axis=1)
        
        # Compute norms
        a_norm = jnp.linalg.norm(a_centered, axis=1)
        b_norm = jnp.linalg.norm(b_centered, axis=1)
        
        # Handle case where norms are zero (all weights equal)
        denominator = a_norm * b_norm
        # Return 0 correlation when denominator is 0 (undefined correlation)
        return jnp.where(denominator == 0, 0.0, numerator / denominator)
    
    # Output: dict with each value being an array of shape [max_steps] containing correlations
    weight_component_correlations = {
        'prev': batch_correlate(w_prev_all, w_final_all),  # [max_steps]
        'incr': batch_correlate(w_incr_all, w_final_all),  # [max_steps]
        'prop': batch_correlate(w_prop_all, w_final_all),  # [max_steps]
        'grid': batch_correlate(w_grid_all, w_final_all),  # [max_steps]
    }
    
    return weight_component_correlations