from typing import NamedTuple, Dict, Any
import jax
import jax.numpy as jnp

from .grid_inference import GridData

from jtap.utils import ChexModelInput
from jtap.utils import JTAPStimulus

# There are so many JTAP types that are defined in the inference package that it makes sense to have a separate file for them.

def jtap_data_to_numpy(jtap_data):
    """
    Recursively converts all JAX arrays in a JTAPData instance to numpy arrays.
    
    Args:
        jtap_data: JTAPData instance containing JAX arrays
        
    Returns:
        JTAPData instance with all JAX arrays converted to numpy arrays
    """
    def convert_recursive(obj):
        if isinstance(obj, jnp.ndarray):
            return jnp.asarray(obj).__array__()
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):  # NamedTuple
            # Get the class type and create new instance with converted fields
            cls = type(obj)
            converted_fields = {}
            for field_name in obj._fields:
                field_value = getattr(obj, field_name)
                converted_fields[field_name] = convert_recursive(field_value)
            return cls(**converted_fields)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_recursive(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: convert_recursive(value) for key, value in obj.items()}
        else:
            # For primitive types (int, float, bool, etc.), return as-is
            return obj
    
    return convert_recursive(jtap_data)

def jtap_data_to_jax(jtap_data):
    """
    Recursively converts all numpy arrays in a JTAPData instance to JAX arrays.
    
    Args:
        jtap_data: JTAPData instance containing numpy arrays
        
    Returns:
        JTAPData instance with all numpy arrays converted to JAX arrays
    """
    def convert_recursive(obj):
        if hasattr(obj, '__array__') and not isinstance(obj, jnp.ndarray):
            # Convert numpy arrays and other array-like objects to JAX arrays
            return jnp.asarray(obj)
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):  # NamedTuple
            # Get the class type and create new instance with converted fields
            cls = type(obj)
            converted_fields = {}
            for field_name in obj._fields:
                field_value = getattr(obj, field_name)
                converted_fields[field_name] = convert_recursive(field_value)
            return cls(**converted_fields)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_recursive(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: convert_recursive(value) for key, value in obj.items()}
        else:
            # For primitive types (int, float, bool, etc.), return as-is
            return obj
    
    return convert_recursive(jtap_data)

class PredictionData(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    speed: jnp.ndarray
    direction: jnp.ndarray
    collision_branch: jnp.ndarray
    rg: jnp.ndarray

class TrackingData(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    direction: jnp.ndarray
    speed: jnp.ndarray

class WeightData(NamedTuple):
    prop_weights: jnp.ndarray
    grid_weights: jnp.ndarray
    incremental_weights: jnp.ndarray
    prev_weights: jnp.ndarray
    final_weights: jnp.ndarray
    incremental_weights_no_obs: jnp.ndarray
    step_prop_weights_regular: jnp.ndarray
    step_prop_weights_alternative: jnp.ndarray

class JTAPParams(NamedTuple):
    max_prediction_steps: int
    max_inference_steps: int
    num_particles: int
    ESS_threshold: float
    simulate_every: int
    inference_input: ChexModelInput

class JTAPInference(NamedTuple):
    tracking: TrackingData
    prediction: PredictionData
    weight_data: WeightData
    grid_data: GridData
    t: int
    resampled: bool
    ESS: float
    is_target_hidden: jnp.ndarray
    is_target_partially_hidden: jnp.ndarray
    obs_is_fully_hidden: bool
    stopped_early: bool

class JTAPData(NamedTuple):
    num_jtap_runs: int
    inference: JTAPInference  # This contains all step data including init
    params: JTAPParams
    step_prop_retvals: Any
    init_prop_retval: Any
    key_seed: int
    stimulus: JTAPStimulus

class JTAPDataAllTrials(NamedTuple):
    runs: Dict[str, JTAPData]