import numpy as np
from scipy.special import logsumexp
from typing import NamedTuple

from jtap_mice.inference import JTAPMiceData

class JTAP_Beliefs(NamedTuple):
    model_beliefs: np.ndarray
    frozen_beliefs: np.ndarray
    decaying_beliefs: np.ndarray
    num_jtap_runs: int

def get_rg_raw_beliefs(JTAP_data, prediction_t_offset):

    assert isinstance(JTAP_data, JTAPMiceData), "JTAP_data must be a JTAPMiceData"
    
    # Get weights and rg data
    weights = JTAP_data.inference.weight_data.final_weights  # shape: (num_jtap_runs, num_timesteps, num_particles) or (num_timesteps, num_particles)
    rg_data = JTAP_data.inference.prediction.rg  # shape: (num_jtap_runs, num_timesteps, max_pred_length, num_particles) or (num_timesteps, max_pred_length, num_particles)
    
    # Handle both single run and multiple runs cases
    if weights.ndim == 2:
        # Single run case: add batch dimension
        weights = weights[None, ...]
        rg_data = rg_data[None, ...]
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    # Normalize weights for each timestep and run
    normalized_probs = np.exp(weights - logsumexp(weights, axis=-1, keepdims=True))  # shape: (num_jtap_runs, num_timesteps, num_particles)
    
    # Get the coded rg hits for the specific prediction offset
    # NOTE: we can check the last timestep simply because the coded rg stays fixed
    # once a sensor has been hit.
    coded_rg_hits = rg_data[:, :, prediction_t_offset - 1, :]  # shape: (num_jtap_runs, num_timesteps, num_particles)
    
    # Compute probabilities for each category across all timesteps and runs
    uncertain_probs = np.sum((coded_rg_hits == 0) * normalized_probs, axis=-1) / np.sum(normalized_probs, axis=-1)
    red_probs = np.sum((coded_rg_hits == 1) * normalized_probs, axis=-1) / np.sum(normalized_probs, axis=-1)
    green_probs = np.sum((coded_rg_hits == 2) * normalized_probs, axis=-1) / np.sum(normalized_probs, axis=-1)
    
    # Stack results
    result = np.stack([green_probs, red_probs, uncertain_probs], axis=-1)  # shape: (num_jtap_runs, num_timesteps, 3)
    
    # Keep batch dimension for consistency - always return (num_jtap_runs, num_timesteps, 3)
    # The calling code expects this shape
    return result


def jtap_baseline_beliefs(model_beliefs, occlusion_bool, decay_T):
    """
    Vectorized numpy implementation of baseline belief decay during occlusion.

    Args:
        model_beliefs: np.ndarray, shape (N, T, 3) or (T, 3) -- model output probabilities
        occlusion_bool: np.ndarray, shape (T,) or (N, T) -- boolean mask for occlusion
        decay_T: float or int -- decay time constant

    Returns:
        np.ndarray, shape (N, T, 3) or (T, 3) -- baseline beliefs with decay applied during occlusion
    """
    model_beliefs = np.asarray(model_beliefs)
    occlusion_bool = np.asarray(occlusion_bool)

    # Normalize shape to (N, T, 3). Some configurations (e.g., single JTAP run or
    # single particle) can introduce extra singleton dimensions. We squeeze
    # non-semantic singleton axes while preserving the final 3-category channel.
    squeeze_batch = False
    if model_beliefs.ndim == 2:
        # (T, 3) -> (1, T, 3)
        assert model_beliefs.shape[1] == 3
        model_beliefs = model_beliefs[None, ...]
        squeeze_batch = True
    elif model_beliefs.ndim == 4:
        # Handle cases like (N, T, T, 3) where there's an extra T dimension
        # This can happen when prediction_t_offset is used incorrectly
        if model_beliefs.shape[-1] == 3:
            # Take the diagonal along the extra T dimension: (N, T, T, 3) -> (N, T, 3)
            if model_beliefs.shape[1] == model_beliefs.shape[2]:
                model_beliefs = np.diagonal(model_beliefs, axis1=1, axis2=2).transpose(0, 2, 1)
            else:
                # If dimensions don't match, take the first slice
                model_beliefs = model_beliefs[:, :, 0, :]
        elif model_beliefs.shape[-2] == 1 and model_beliefs.shape[-1] == 3:
            model_beliefs = model_beliefs[:, :, 0, :]
        elif model_beliefs.shape[-2] == 3 and model_beliefs.shape[-1] == 1:
            model_beliefs = model_beliefs[:, :, :, 0]
        else:
            squeezed = np.squeeze(model_beliefs)
            if squeezed.ndim == 3 and squeezed.shape[-1] == 3:
                model_beliefs = squeezed
            else:
                raise ValueError(
                    f"Unexpected belief array shape {model_beliefs.shape}; expected (N,T,3) up to singleton dims"
                )

    N, T, C = model_beliefs.shape
    assert C == 3

    # Broadcast occlusion_bool if needed
    if occlusion_bool.ndim == 1:
        occlusion_bool = np.broadcast_to(occlusion_bool, (N, T))
    elif occlusion_bool.shape == (T,):
        occlusion_bool = np.broadcast_to(occlusion_bool, (N, T))
    elif occlusion_bool.shape == (N, T):
        pass
    else:
        raise ValueError(f"occlusion_bool must be shape (T,) or (N, T), got {occlusion_bool.shape}")

    # Find the start of each occlusion run
    occl_starts = np.zeros_like(occlusion_bool, dtype=bool)
    occl_starts[:, 1:] = occlusion_bool[:, 1:] & (~occlusion_bool[:, :-1])
    occl_starts[:, 0] = occlusion_bool[:, 0]

    # For each occlusion run, assign a unique id (increment at each start)
    occl_run_ids = np.cumsum(occl_starts, axis=1) * occlusion_bool

    # Get the first index of each run for each batch
    first_occl_idx = np.zeros_like(occl_run_ids)
    for n in range(N):
        run_ids = occl_run_ids[n]
        idxs = np.arange(T)
        # For each unique run id (excluding 0), find the first index
        unique_ids = np.unique(run_ids[run_ids > 0])
        for rid in unique_ids:
            first_idx = np.argmax((run_ids == rid))
            first_occl_idx[n, run_ids == rid] = first_idx

    # The occlusion run length at each (n, t) is t - first_occl_idx[n, t] + 1 if occluded, else 0
    occl_len = np.where(occlusion_bool, np.arange(T)[None, :] - first_occl_idx + 1, 0)

    # For each (n, t), we need to know the "frozen" belief at the start of the occlusion run
    frozen_beliefs = np.take_along_axis(
        model_beliefs,
        first_occl_idx[..., None],
        axis=1
    )  # shape (N, T, 3)

    # Compute decay rate
    decay_rate = np.log(2) / decay_T if decay_T != np.inf else 0.0

    # For all occluded positions, apply exponential decay to non-target indices (0,1)
    non_target = np.array([0, 1])
    decayed = frozen_beliefs[..., non_target] * np.exp(-decay_rate * occl_len[..., None])
    increased = 1.0 - np.sum(decayed, axis=-1, keepdims=True)
    new_probs = np.zeros_like(model_beliefs)
    new_probs[..., non_target] = decayed
    new_probs[..., 2:3] = increased

    # Output: for occluded, use new_probs; for not occluded, use model_beliefs
    out = np.where(occlusion_bool[..., None], new_probs, model_beliefs)

    # Always set t=0 to model_beliefs (no decay at first frame)
    out[:, 0, :] = model_beliefs[:, 0, :]

    # Remove batch dimension if input was (T, 3)
    if squeeze_batch:
        out = out[0]

    return out

def jtap_compute_beliefs(_jtap_data_, pred_len = None, decay_T = 20, partial_occlusion_counts_as_occlusion = True):

    if isinstance(_jtap_data_, JTAPMiceData):
        is_multiple_runs = _jtap_data_.num_jtap_runs > 1
        # max_prediction_steps is always a list, so we need to extract the scalar value
        max_prediction_steps = _jtap_data_.params.max_prediction_steps[0] if isinstance(_jtap_data_.params.max_prediction_steps, (list, np.ndarray)) else _jtap_data_.params.max_prediction_steps
    
        if pred_len is not None:
            # first ensure pred_len is not greater than max_prediction_steps
            if max_prediction_steps < pred_len:
                raise ValueError(f"pred_len is greater than max_prediction_steps: {max_prediction_steps} < {pred_len}")
            # then do a min in case pred_len is less than max_prediction_steps
            pred_len = np.minimum(pred_len, max_prediction_steps)
        else:
            # if pred_len is not provided, use max_prediction_steps
            pred_len = max_prediction_steps
        # get raw beliefs
        model_beliefs = get_rg_raw_beliefs(_jtap_data_, pred_len)
        # get occlusion bool depending on whether partial occlusion counts as occlusion
        if partial_occlusion_counts_as_occlusion:
            occlusion_bool = _jtap_data_.stimulus.fully_occluded_bool | _jtap_data_.stimulus.partially_occluded_bool 
        else:
            occlusion_bool = _jtap_data_.stimulus.fully_occluded_bool 

        # get frozen beliefs
        frozen_beliefs = jtap_baseline_beliefs(model_beliefs, occlusion_bool, np.inf)
        # get decaying beliefs
        decaying_beliefs = jtap_baseline_beliefs(model_beliefs, occlusion_bool, decay_T)
        
        jtap_beliefs = JTAP_Beliefs(model_beliefs = model_beliefs, frozen_beliefs = frozen_beliefs, decaying_beliefs = decaying_beliefs, num_jtap_runs = _jtap_data_.num_jtap_runs)
        return jtap_beliefs
    else:
        raise ValueError(f"Unsupported type: {type(_jtap_data_)}, supported types are JTAPMiceData")