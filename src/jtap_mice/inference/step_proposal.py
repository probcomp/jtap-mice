from typing import NamedTuple, Dict, Any
import jax
import jax.numpy as jnp
import genjax
from genjax import gen, ChoiceMap
from genjax import ChoiceMapBuilder as C

from jtap_mice.distributions import truncatednormposition2d, circular_normal
from jtap_mice.utils.common_math import angle_diff_radians

class TopDownData(NamedTuple):
    top_down_next_direction: jnp.ndarray
    top_down_next_speed: jnp.ndarray
    top_down_next_x: jnp.ndarray
    top_down_next_y: jnp.ndarray
    current_x: jnp.ndarray
    current_y: jnp.ndarray
    current_vx: jnp.ndarray
    current_vy: jnp.ndarray
    current_speed: jnp.ndarray
    current_direction: jnp.ndarray

class ProposalOutlierData(NamedTuple):
    proposal_outlier_prob: jnp.ndarray
    proposal_is_outlier: jnp.ndarray

class StepProposalRetval(NamedTuple):
    step_prop_t: jnp.int32
    proposal_outlier_data: ProposalOutlierData
    bottom_up_speed_mean: jnp.ndarray
    bottom_up_direction_mean: jnp.ndarray
    mean_vx_since_last_collision: jnp.ndarray
    mean_vy_since_last_collision: jnp.ndarray
    last_collision_T: jnp.ndarray
    last_collision_x: jnp.ndarray       
    last_collision_y: jnp.ndarray
    bottom_up_proposed_x: jnp.ndarray
    bottom_up_proposed_y: jnp.ndarray
    step_prop_x: jnp.ndarray
    step_prop_y: jnp.ndarray
    top_down_will_collide: jnp.ndarray
    step_prop_speed: jnp.ndarray
    step_prop_direction: jnp.ndarray
    use_top_down_proposal: jnp.ndarray
    use_hybrid_proposal: jnp.ndarray
    use_bottom_up_proposal: jnp.ndarray
    top_down_data: TopDownData
    prob_use_bottom_up: jnp.ndarray

@gen
def top_down_step_proposal(mi, top_down_next_speed, top_down_next_direction, top_down_next_x, top_down_next_y, col_branch):
    """
    Pure top-down proposal when object is fully hidden.
    
    Mean values:
    - Position (x, y): Obtained by running the dynamics model forward from the previous inferred state
    - Direction: Obtained by running the dynamics model forward from the previous inferred state  
    - Speed: Obtained by running the dynamics model forward from the previous inferred state
    
    Sampling strategy:
    - Position (x, y): Used deterministically from dynamics model output (no additional noise)
    - Direction: Sampled with circular normal noise around dynamics model output
      - Uses σ_COL_direction_occ if collision occurred (col_branch == 4)
      - Uses σ_NOCOL_direction_occ otherwise
    - Speed: Sampled with truncated normal noise around dynamics model output using σ_speed_stepprop
    """

    direction_noise = jnp.where(col_branch == jnp.float32(4), mi.σ_NOCOL_direction_stepprop, mi.σ_COL_direction_prop)
    speed_noise = mi.σ_speed_stepprop

    proposed_speed = genjax.truncated_normal(top_down_next_speed, speed_noise, jnp.float32(0), mi.max_speed) @ "speed"
    proposed_direction = circular_normal(top_down_next_direction, direction_noise) @ "direction"
    return proposed_speed, proposed_direction, top_down_next_x, top_down_next_y

@gen
def bottom_up_step_proposal(mi, bottom_up_speed_mean, bottom_up_direction_mean, bottom_up_proposed_x, bottom_up_proposed_y, t):
    """
    Pure bottom-up proposal when object is visible and not expected to have collided this timestep.
    
    Mean values:
    - Position (x, y): Obtained from positional grid inference (already sampled during grid inference)
    - Direction: Derived from proposed position by calculating trajectory from last collision point
    - Speed: Derived from proposed position by calculating trajectory from last collision point
    
    Sampling strategy:
    - Position (x, y): Sampled from grid inference output (no additional noise)
    - Direction: Sampled with circular normal noise around derived direction mean
      - Uses σ_NOCOL_direction_initprop for early timesteps (t <= 2)
      - Uses σ_NOCOL_direction_stepprop for later timesteps
    - Speed: Sampled with truncated normal noise around derived speed mean
      - Uses σ_speed_initprop for early timesteps (t <= 2)
      - Uses σ_speed_stepprop for later timesteps
    """

    proposed_direction_noise = jax.lax.select(
        jnp.less_equal(t, jnp.int32(2)),
        mi.σ_NOCOL_direction_initprop,
        mi.σ_NOCOL_direction_stepprop
    )

    proposed_speed_noise = jax.lax.select(
        jnp.less_equal(t, jnp.int32(2)),
        mi.σ_speed_initprop,
        mi.σ_speed_stepprop
    )
    proposed_speed = genjax.truncated_normal(bottom_up_speed_mean, proposed_speed_noise, jnp.float32(0), mi.max_speed) @ "speed"
    proposed_direction = circular_normal(bottom_up_direction_mean, proposed_direction_noise) @ "direction"
    return proposed_speed, proposed_direction, bottom_up_proposed_x, bottom_up_proposed_y

step_proposal_switch = genjax.switch(
    top_down_step_proposal,
    bottom_up_step_proposal
)


@gen
def step_proposal(mi, is_fully_hidden, num_target_pixels, mo, bottom_up_proposed_x, prev_prediction_data, step_prop_t):

    # extract current data for saving
    current_x, current_y, current_direction, current_speed = mo.x, mo.y, mo.direction, mo.speed
    current_vx, current_vy = mo.speed * jnp.cos(mo.direction), mo.speed * jnp.sin(mo.direction)
    # ============================================================================
    # BOTTOM-UP COMPUTATIONS (from observations/grid inference)
    # ============================================================================

    # NOTE: an alternative is to use the data_driven_size_and_position function to get the mean values instead of grid inference
    # _, bottom_up_proposed_x, bottom_up_proposed_y, _ = data_driven_size_and_position(step_obs, mi.image_discretization)

    # clip grid proposed x and y to range
    bottom_up_proposed_x = jnp.clip(bottom_up_proposed_x, jnp.float32(0), mi.scene_dim[0] - mo.diameter)
    bottom_up_proposed_y = jnp.clip(bottom_up_proposed_y, jnp.float32(0), mi.scene_dim[1] - mo.diameter)

    last_collision_T, last_collision_x, last_collision_y = mo.last_collision_data[:3]

    # # velocity gradient since last collision
    mean_vx_since_last_collision, mean_vy_since_last_collision = ((bottom_up_proposed_x - last_collision_x)/(step_prop_t - last_collision_T), (bottom_up_proposed_y - last_collision_y)/(step_prop_t - last_collision_T))

    # mean_vx_since_last_collision = bottom_up_proposed_x - current_x
    # mean_vy_since_last_collision = bottom_up_proposed_y - current_y
    
    bottom_up_speed_mean = jnp.sqrt(jnp.square(mean_vx_since_last_collision) + jnp.square(mean_vy_since_last_collision))
    bottom_up_speed_mean = jnp.clip(bottom_up_speed_mean, jnp.float32(0) + jnp.finfo(float).eps, mi.max_speed - jnp.finfo(float).eps)
    # CLIP SPEED TO WITHIN MAX SPEED --> otherwise it will propose somethign absurd and will 
    # get a very low logprob --> high particle weight in turn (p/q)
    bottom_up_direction_mean = jnp.arctan2(mean_vy_since_last_collision, mean_vx_since_last_collision)


    # ============================================================================
    # TOP-DOWN COMPUTATIONS (from dynamics model --via simulations)
    # ============================================================================

    # get the timestep from the simulation that corresponds to the current step
    t_from_simulation = step_prop_t % mi.simulate_every

    top_down_next_x = prev_prediction_data.x[t_from_simulation]
    top_down_next_y = prev_prediction_data.y[t_from_simulation]
    top_down_next_speed = prev_prediction_data.speed[t_from_simulation]
    top_down_next_direction = prev_prediction_data.direction[t_from_simulation]
    col_branch = prev_prediction_data.collision_branch[t_from_simulation]
    top_down_will_collide = jnp.not_equal(col_branch, jnp.float32(4.))

    # NOTE: BREAKING CHANGE --> use the current speed and direction of the object instead of the predicted speed and direction as means for the top-down proposal
    # NOTE: what this should really be -- is that the speed and direction use
    # predictions but behave as if they were sampled from the proposal
    # I have to figure this out
    top_down_next_speed_BOTTOM_UP = jnp.where(top_down_will_collide, top_down_next_speed, mo.speed)
    top_down_next_direction_BOTTOM_UP = jnp.where(top_down_will_collide, top_down_next_direction, mo.direction)

    # clip x and y positions to range
    top_down_next_x = jnp.clip(top_down_next_x, jnp.float32(0), mi.scene_dim[0] - mo.diameter)
    top_down_next_y = jnp.clip(top_down_next_y, jnp.float32(0), mi.scene_dim[1] - mo.diameter)

    # ============================================================================
    # PROPOSAL STRATEGY SELECTION AND EXECUTION
    # ============================================================================

    # NOTE: hardcoded number of blue pixels is 80
    prob_use_bottom_up = num_target_pixels / 80

    use_bottom_up_proposal = genjax.flip(prob_use_bottom_up) @ "use_bottom_up_proposal"

    use_top_down_proposal = jnp.logical_not(use_bottom_up_proposal)

    # # use top down proposal if object is fully hidden
    # use_top_down_proposal = is_fully_hidden

    # use hybrid proposal if object is visible and expected to collide
    use_hybrid_proposal = jnp.logical_and(
        jnp.greater(step_prop_t,jnp.int32(1)),
        top_down_will_collide
    )

    # use bottom up proposal if object is visible and not expected to collide
    # use_bottom_up_proposal = jnp.logical_not(
    #     jnp.logical_or(
    #         use_top_down_proposal,
    #         use_hybrid_proposal
    #     )
    # )

    switch_branch = jnp.where(
        use_top_down_proposal, # this step is probabilistic
        jnp.int32(0),
        jnp.where(
            use_hybrid_proposal, # this step is deterministic
            jnp.int32(1),
            jnp.int32(2)
        )
    )
    

    step_prop_speed, step_prop_direction, step_prop_x, step_prop_y = step_proposal_switch(switch_branch, 
        (mi, top_down_next_speed_BOTTOM_UP, top_down_next_direction_BOTTOM_UP, top_down_next_x, top_down_next_y, col_branch),
        (mi, top_down_next_speed, top_down_next_direction, bottom_up_proposed_x, bottom_up_proposed_y),
        (mi, bottom_up_speed_mean, bottom_up_direction_mean, bottom_up_proposed_x, bottom_up_proposed_y, step_prop_t)
    ) @ 'prop_switch' 

    # in this proposal, we actively compute the outlier probability
    # ============================================================================
    # PROBABILISTIC OUTLIER STEP DETECTION
    # ============================================================================

    # STEP 1
    # I want to get a sense of how different the proposed direction is away from the predicted direction
    # which basically compares top_down_next_direction with step_prop_direction
    # NOTE: That we only expect this to happen when a bottom up proposal drives the direction in a different direction

    proposal_outlier_prob = proposal_direction_outlier_prob(mi, top_down_next_direction, step_prop_direction)

    proposal_is_outlier = genjax.flip(proposal_outlier_prob) @ "proposal_is_outlier"

    proposal_outlier_data = ProposalOutlierData(
        proposal_outlier_prob=proposal_outlier_prob,
        proposal_is_outlier=proposal_is_outlier
    )

    top_down_data = TopDownData(
        top_down_next_direction=top_down_next_direction,
        top_down_next_speed=top_down_next_speed,
        top_down_next_x=top_down_next_x,
        top_down_next_y=top_down_next_y,
        current_x=current_x,
        current_y=current_y,
        current_vx=current_vx,
        current_vy=current_vy,
        current_speed=current_speed,
        current_direction=current_direction
    )

    return StepProposalRetval(
        proposal_outlier_data=proposal_outlier_data,
        step_prop_t=step_prop_t,
        bottom_up_speed_mean=bottom_up_speed_mean,
        bottom_up_direction_mean=bottom_up_direction_mean,
        mean_vx_since_last_collision=mean_vx_since_last_collision,
        mean_vy_since_last_collision=mean_vy_since_last_collision,
        last_collision_T=last_collision_T,
        last_collision_x=last_collision_x,
        last_collision_y=last_collision_y,
        bottom_up_proposed_x=bottom_up_proposed_x,
        bottom_up_proposed_y=bottom_up_proposed_y,
        step_prop_x=step_prop_x,
        step_prop_y=step_prop_y,
        top_down_will_collide=top_down_will_collide,
        step_prop_speed=step_prop_speed,
        step_prop_direction=step_prop_direction,
        use_top_down_proposal=use_top_down_proposal,
        use_hybrid_proposal=use_hybrid_proposal,
        use_bottom_up_proposal=use_bottom_up_proposal,
        top_down_data=top_down_data,
        prob_use_bottom_up=prob_use_bottom_up
    )

def proposal_direction_outlier_prob(mi, top_down_next_direction, step_prop_direction):
    angular_diff = angle_diff_radians(top_down_next_direction, step_prop_direction)
    sigmoid_input = mi.proposal_direction_outlier_alpha * (jnp.abs(angular_diff) - mi.proposal_direction_outlier_tau)
    return jax.nn.sigmoid(sigmoid_input)

# def step_choicemap_translator(cm_data_driven_proposal, cm_grid_inference_proposal, cm_obs):
#     return C.d(
#             {
#                 'step' : cm_data_driven_proposal('prop_switch').merge(cm_grid_inference_proposal).merge(cm_data_driven_proposal['is_outlier_step']),
#                 'obs' : cm_obs['obs']
#             }
#         )

def step_choicemap_translator(
    cm_data_driven_proposal,
    cm_grid_inference_proposal,
    cm_obs
):
    """
    Translates and merges choicemaps for the step proposal,
    grid inference proposal, and observation.
    """
    step_cm = (
        cm_data_driven_proposal('prop_switch')
        .merge(cm_grid_inference_proposal)
        .merge(
            C['is_outlier_step'].set(
                cm_data_driven_proposal['proposal_is_outlier']
            )
        )
    )
    obs_cm = cm_obs['obs']
    return C.d({
        'step': step_cm,
        'obs': obs_cm,
    })