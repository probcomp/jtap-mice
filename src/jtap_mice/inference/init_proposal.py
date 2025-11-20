from typing import NamedTuple
import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from genjax import ChoiceMapBuilder as C

from .data_driven import data_driven_size_and_position

from jtap_mice.distributions import truncatednormposition2d

class InitProposalRetval(NamedTuple):
    init_prop_x: jnp.ndarray
    data_driven_x: jnp.ndarray

@gen
def init_proposal(init_proposal_args):
    """
    This function proposes a new initial position for the ball with the assumption it is visible.
    """
    mi, initial_discrete_obs = init_proposal_args

    pos_noise = mi.σ_pos_initprop
    scale = mi.image_discretization
    diameter = mi.diameter

    
    _, data_driven_x, *_ = data_driven_size_and_position(initial_discrete_obs, scale)

    init_prop_x = genjax.truncated_normal(data_driven_x, pos_noise, -diameter, mi.scene_dim[0]) @ "x"
    
    return InitProposalRetval(
        init_prop_x=init_prop_x,
        data_driven_x=data_driven_x,
    )

@jax.jit
def init_choicemap_translator(cm_proposed, init_discrete_obs):

    return C.d(
            {
                'init' : cm_proposed,
                'init_obs' : init_discrete_obs
            }
        )
