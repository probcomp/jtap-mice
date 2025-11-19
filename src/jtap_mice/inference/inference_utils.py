import numpy as np

# to keep observations a fixed size for the sake of JITTed inference
def pad_obs_with_last_frame(observations, target_num_frames):
    """
    Pad an observation sequence to a target length by repeating the last frame.
    
    This function extends a sequence of observations to a fixed length by duplicating
    the final frame. This is useful for creating fixed-size arrays for JIT compilation
    while preserving the temporal structure of the original sequence.
    
    Args:
        observations: Array of shape (num_frames, ...) containing the observation sequence
        target_num_frames: Target length for the padded sequence
        
    Returns:
        Array of shape (target_num_frames, ...) with the original observations followed
        by repeated copies of the last frame
        
    Raises:
        AssertionError: If target_num_frames is not greater than the current sequence length
    """
    assert target_num_frames >= observations.shape[0], "target_num_frames must be greater than the number of frames in the array"
    
    num_padding_frames = target_num_frames - observations.shape[0]
    last_frame = observations[-1]
    padding_frames = np.repeat(last_frame[np.newaxis, ...], num_padding_frames, axis=0)
    padded_observations = np.concatenate([observations, padding_frames], axis=0)
    
    return padded_observations