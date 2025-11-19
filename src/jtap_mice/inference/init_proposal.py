from typing import NamedTuple
import jax
import jax.numpy as jnp
from genjax import gen
from genjax import ChoiceMapBuilder as C

from .data_driven import data_driven_size_and_position

from jtap_mice.distributions import truncatednormposition2d

class InitProposalRetval(NamedTuple):
    init_prop_x: jnp.ndarray
    init_prop_y: jnp.ndarray
    data_driven_x: jnp.ndarray
    data_driven_y: jnp.ndarray

@gen
def init_proposal(init_proposal_args):
    """
    This function proposes a new initial position for the ball with the assumption it is visible.
    """
    mi, initial_discrete_obs = init_proposal_args

    pos_noise = mi.σ_pos_initprop
    scale = mi.image_discretization
    _, data_driven_x, data_driven_y, *_ = data_driven_size_and_position(initial_discrete_obs, scale)

    init_prop_x, init_prop_y = truncatednormposition2d((data_driven_x, pos_noise, 0., mi.scene_dim[0] - mi.diameter), 
        (data_driven_y, pos_noise, 0.,  mi.scene_dim[1] - mi.diameter)) @ "xy" # joint sampling of x and y
    
    return InitProposalRetval(
        init_prop_x=init_prop_x,
        init_prop_y=init_prop_y,
        data_driven_x=data_driven_x,
        data_driven_y=data_driven_y
    )

@jax.jit
def init_choicemap_translator(cm_proposed, init_discrete_obs):

    return C.d(
            {
                'init' : cm_proposed,
                'init_obs' : init_discrete_obs
            }
        )
