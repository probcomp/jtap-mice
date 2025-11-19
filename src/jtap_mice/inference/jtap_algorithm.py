
import genjax
import jax
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
from genjax import ChoiceMapBuilder as C

from .data_driven import data_driven_size_and_position
from .init_proposal import init_proposal, init_choicemap_translator
from .step_proposal import step_proposal, step_choicemap_translator
from .grid_inference import grid_proposer, GridData, grid_likelihood_evaluator, make_position_grid, find_valid_positions_bool, adaptive_grid_size
from .jtap_types import JTAPMiceData, JTAPParams, JTAPInference, JTAPMiceDataAllTrials, PredictionData, TrackingData, WeightData, jtap_data_to_numpy
from .inference_utils import pad_obs_with_last_frame

from jtap_mice.model import full_init_model, full_step_model, stepper_model, red_green_sensor_readouts, stepper_model_no_obs
from jtap_mice.utils import effective_sample_size, init_step_concat, slice_pt, multislice_pytree

# Define all the jitted functions
init_proposer= jax.vmap(init_proposal.propose, in_axes = (0,None))
init_choicemap_merger = jax.vmap(init_choicemap_translator, in_axes = (0, None))
step_proposer= jax.vmap(step_proposal.propose, in_axes = (0,(None,None,None,0,0,0,1,None)))
step_proposal_assessor = jax.vmap(step_proposal.assess, in_axes = (0,(None,None,None,0,0,0,1,None)))
step_choicemap_merger = jax.vmap(step_choicemap_translator, in_axes = (0,0,None))
initializer = jax.vmap(full_init_model.importance, in_axes = (0,0,None))
stepper = jax.vmap(full_step_model.assess, in_axes = (0,(0,None,None)))
stepper_no_obs = jax.vmap(stepper_model_no_obs.assess, in_axes = (0,(0,None,None)))
simulator = jax.vmap(stepper_model.simulate, in_axes = (0,(0,None,None)))
idx_resampler = jax.vmap(genjax.categorical.simulate, in_axes=(0, None))
mo_resampler = jax.vmap(lambda idx, mos:jtu.tree_map(lambda v : v[idx], mos), in_axes = (0, None))

# Actual JTAP SMC + Prediction code
@partial(jax.jit, static_argnames=["max_inference_steps", "num_particles"])
def run_jtap_(initial_key, mi, ESS_proportion, discrete_obs, max_inference_steps, num_particles):

    # while max num inference steps can be equal to the number of observed frames,
    # we keep the option to run less than that

    def resample_fn(weights, step_mos, resample_key):
        idxs = idx_resampler(jax.random.split(resample_key, num_particles), (weights,)).get_retval()
        mos_resampled = mo_resampler(idxs, step_mos)
        return mos_resampled, jnp.zeros_like(weights)

    def prediction(carry, _):
        key, sim_mos, coded_rg = carry
        next_key, sim_key = jax.random.split(key, 2)
        sim_keys = jax.random.split(sim_key, num_particles)
        inference_mode_bool = jnp.bool_(False) # ONLY SIMULATION MODE
        next_sim_mos = simulator(sim_keys, (sim_mos, mi, inference_mode_bool)).get_retval()
        stopped_early = jnp.any(next_sim_mos.stopped_early)
        # the coded rg converts to the sensor reading during the timestep it hits the sensor
        # and stays static for the rest of the prediction steps. This way, we can just 
        # look at the last timestep to see which it hit (if it hit any)
        coded_rg = jnp.where(coded_rg == jnp.int8(0), red_green_sensor_readouts(next_sim_mos), coded_rg)
        prediction_data = PredictionData(x=next_sim_mos.x, y=next_sim_mos.y, speed=next_sim_mos.speed, direction=next_sim_mos.direction, collision_branch=next_sim_mos.collision_branch, rg=coded_rg)
        return (next_key, next_sim_mos, coded_rg), (prediction_data, stopped_early)

    def particle_filter_step_rg(carry, t):
        key, mos, weights, prev_num_obj_pixels, prev_prediction_data = carry

        SINGLE_MO = slice_pt(mos,0)

        simulate_this_step = jnp.equal(t % mi.simulate_every, 0)

        # Split main keys
        next_key, sim_key, stepper_key, proposal_key, resample_key = jax.random.split(key, 5)
        data_driven_proposal_key, grid_proposal_key = jax.random.split(proposal_key, 2)
        grid_sampling_key, _ = jax.random.split(grid_proposal_key, 2)
        # vmap keys
        stepper_keys = jax.random.split(stepper_key, num_particles)
        data_driven_proposal_keys = jax.random.split(data_driven_proposal_key, num_particles)
        grid_sampling_keys = jax.random.split(grid_sampling_key, num_particles)

        # get obs choicemap
        step_obs_chm = C.d({
                "obs": discrete_obs[t],
            })
        ##################################################################################
        # GRID INFERENCE (STEP)
        ##################################################################################
        _, x_grid_center, y_grid_center, obs_is_fully_hidden, num_obj_pixels = data_driven_size_and_position(discrete_obs[t], mi.image_discretization)

        grid_size = adaptive_grid_size(num_obj_pixels, mi.grid_size_bounds)

        # step model args
        inference_mode_bool = jnp.bool_(True) # INFERENCE MODE
        step_model_args = (mos, mi, inference_mode_bool)

        #TODO: GRIDS must NOT allow positions stuck inside barriers
        position_grid, pos_cells = make_position_grid(mi, x_grid_center, y_grid_center, grid_size, grid_size)
        valid_position_bools = find_valid_positions_bool(position_grid, SINGLE_MO.diameter, SINGLE_MO.masked_barriers) 

        # EVALUATE GRID SCORES
        total_grid_size = pos_cells.shape[0]
        observed_image = step_obs_chm['obs']
        grid_render_args = (
            mi.pix_x, mi.pix_y, mi.diameter,
            position_grid['x'], position_grid['y'], SINGLE_MO.masked_barriers, SINGLE_MO.masked_occluders,
            SINGLE_MO.red_sensor, SINGLE_MO.green_sensor, mi.image_discretization
        )

        # Pass all required arguments to grid_likelihood_evaluator, matching model.py
        # grid_logprobs = jnp.zeros((36,))
        grid_logprobs = grid_likelihood_evaluator(
            observed_image,
            grid_render_args,
            mi.pixel_corruption_prob,
            mi.tile_size_arr,
            mi.σ_pixel_spatial,
            mi.image_power_beta
        )
        grid_logprobs = jnp.where(valid_position_bools, grid_logprobs, jnp.array(-jnp.inf))

        # SAMPLE GRID and CONTINUOUS SAMPLE
        _, grid_weights, grid_retvals = grid_proposer(grid_sampling_keys, (grid_logprobs, pos_cells))
        (grid_proposed_xs, grid_proposed_ys, sampled_idxs, sampled_x_cells, sampled_y_cells) = grid_retvals

        
        grid_data = GridData(
            grid_proposed_xs=grid_proposed_xs, # final proposed x
            grid_proposed_ys=grid_proposed_ys, # final proposed y
            sampled_idxs=sampled_idxs, # sampled grid index
            sampled_x_cells=sampled_x_cells, # sampled x cell
            sampled_y_cells=sampled_y_cells, # sampled y cell
            grid_size=grid_size, # full grid size
            x_grid_center=x_grid_center, # centre of grid
            y_grid_center=y_grid_center, # centre of grid
            x_grid=position_grid['x'], # x grid
            y_grid=position_grid['y'], # y grid
            grid_logprobs=grid_logprobs, # grid logprobs
            num_obj_pixels=num_obj_pixels * jnp.ones((num_particles,)), # number of object pixels
        )

        ##################################################################################
        # PROPOSAL + UPDATE INFERENCE (STEP)
        ##################################################################################

        # Propose new step particle extensions
        step_proposal_args = (mi, obs_is_fully_hidden, num_obj_pixels, mos, grid_proposed_xs, grid_proposed_ys, prev_prediction_data, t)
        step_proposed_choices, step_prop_weights_regular, step_prop_retval = step_proposer(data_driven_proposal_keys, step_proposal_args)

        # get the weights of the alternative switch branch since this is a mixture proposal
        use_bottom_up_proposal = step_prop_retval.use_bottom_up_proposal
        use_top_down_proposal = step_prop_retval.use_top_down_proposal
        alternative_use_bottom_up_proposal = jnp.logical_not(use_bottom_up_proposal)
        alternative_proposal_choices = step_proposed_choices.at['use_bottom_up_proposal'].set(alternative_use_bottom_up_proposal)
        alternative_proposal_weights, _ = step_proposal_assessor(alternative_proposal_choices, step_proposal_args)

        # logsumpexp the two proposal weights
        step_prop_weights = jax.nn.logsumexp(jnp.stack([step_prop_weights_regular, alternative_proposal_weights], axis = 0), axis = 0)

        # Merge choicemap
        step_prop_xs, step_prop_ys = step_prop_retval.step_prop_x, step_prop_retval.step_prop_y

        # SWITCH CHM for Occlusion
        position_choices = jax.vmap(lambda x,y: C['x'].set(x).at['y'].set(y))(step_prop_xs, step_prop_ys)

        # first evaluate the weights for the non-outlier target model
        step_inference_chm_merged = step_choicemap_merger(step_proposed_choices, position_choices, step_obs_chm)
        incremental_weights, step_mos = stepper(step_inference_chm_merged, step_model_args)

        # # do the same for the no-obs weights (used for inspection only, not in inference)
        # incremental_weights_no_obs, _ = stepper_no_obs(step_inference_chm_merged('step'), step_model_args)
        incremental_weights_no_obs = jnp.zeros_like(incremental_weights)

        grid_weights = jnp.where(use_top_down_proposal, jnp.zeros_like(grid_weights), grid_weights)

        # Update weights
        final_weights = weights + incremental_weights - step_prop_weights - grid_weights
        
        weight_data = WeightData(
            prop_weights=step_prop_weights,
            grid_weights=grid_weights,
            incremental_weights=incremental_weights,
            prev_weights=weights,
            final_weights=final_weights,
            incremental_weights_no_obs=incremental_weights_no_obs,
            step_prop_weights_regular=step_prop_weights_regular,
            step_prop_weights_alternative=alternative_proposal_weights,
        )
        
        # Resample if ESS is low
        ESS = effective_sample_size(final_weights)
        
        step_mos, final_weights_post_resample = jax.lax.cond(
            ESS < ESS_threshold,
            resample_fn,
            lambda *_ : (step_mos, final_weights),
            final_weights, step_mos, resample_key
        )

        prediction_data, pred_stopped_early = jax.lax.cond(
            simulate_this_step,
            lambda: jax.lax.scan(
                prediction,
                (sim_key, step_mos, red_green_sensor_readouts(step_mos)),
                jnp.arange(max_prediction_steps)
            )[1],
            lambda: (prev_prediction_data, jnp.zeros(max_prediction_steps, dtype=jnp.bool_))
        )

        stopped_early = jnp.logical_or(jnp.any(step_mos.stopped_early), jnp.any(pred_stopped_early))

        tracking_data = TrackingData(
            x=step_mos.x,
            y=step_mos.y,
            direction=step_mos.direction,
            speed=step_mos.speed
        )
        
        JTAP_data_step = JTAPInference(
            tracking=tracking_data,
            prediction=prediction_data,
            weight_data=weight_data,
            grid_data=grid_data,
            t=t,
            resampled=ESS < ESS_threshold,
            ESS=ESS,
            is_target_hidden=step_mos.is_target_hidden,
            is_target_partially_hidden=step_mos.is_target_partially_hidden,
            obs_is_fully_hidden=obs_is_fully_hidden,
            stopped_early=stopped_early
        )
        
        return (next_key, step_mos, final_weights_post_resample, num_obj_pixels, prediction_data), (JTAP_data_step, step_prop_retval, None)

    # Prepare initial carry state
    max_prediction_steps = max_inference_steps + 5 # fix to max num inference steps
    ESS_threshold = ESS_proportion * num_particles
    key, subkey, smc_key, sim_key = jax.random.split(initial_key,4)
    initializer_keys = jax.random.split(key, num_particles)
    proposal_keys = jax.random.split(subkey, num_particles)

    # only using this to get obs_is_fully_hidden
    *_, obs_is_fully_hidden, num_obj_pixels = data_driven_size_and_position(discrete_obs[0], mi.image_discretization)

    # Initial proposal
    init_proposal_args = ((mi, discrete_obs[0]),)
    init_proposed_choices, init_prop_weights, init_prop_retval = init_proposer(proposal_keys, init_proposal_args)    
    init_inference_chm_merged = init_choicemap_merger(init_proposed_choices, discrete_obs[0])
    model_args = (mi,)

    init_particles, init_weights = initializer(initializer_keys, init_inference_chm_merged, model_args)
    initial_weights = init_weights - init_prop_weights
    init_mos = init_particles.get_retval()

    _, (init_prediction_data, pred_stopped_early) = jax.lax.scan(
            prediction,
            (sim_key, init_mos, red_green_sensor_readouts(init_mos)),
            jnp.arange(max_prediction_steps)
        )
    
    stopped_early = jnp.logical_or(jnp.any(init_mos.stopped_early), jnp.any(pred_stopped_early))

    init_tracking_data = TrackingData(
        x=init_mos.x,
        y=init_mos.y,
        direction=init_mos.direction,
        speed=init_mos.speed
    )

    # Create empty grid data for initial step with correct shapes
    total_grid_size = mi.num_x_grid_arr.shape[0] * mi.num_y_grid_arr.shape[0]
    init_grid_data = GridData(
        grid_proposed_xs=jnp.zeros((num_particles,)),
        grid_proposed_ys=jnp.zeros((num_particles,)),
        sampled_idxs=jnp.zeros((num_particles,), dtype=jnp.int32),
        sampled_x_cells=jnp.zeros((num_particles, 2)),
        sampled_y_cells=jnp.zeros((num_particles, 2)),
        grid_size=jnp.float32(0.0),
        x_grid_center=jnp.float32(0.0),
        y_grid_center=jnp.float32(0.0),
        x_grid=jnp.zeros((total_grid_size,)),
        y_grid=jnp.zeros((total_grid_size,)),
        grid_logprobs=jnp.zeros((total_grid_size,)),
        num_obj_pixels=num_obj_pixels * jnp.ones((num_particles,))
    )

    init_weight_data = WeightData(
        prop_weights=init_prop_weights,
        grid_weights=jnp.zeros_like(init_prop_weights),
        incremental_weights=init_weights,
        prev_weights=jnp.zeros_like(initial_weights),
        final_weights=initial_weights,
        incremental_weights_no_obs=jnp.zeros_like(init_weights),
        step_prop_weights_regular=init_prop_weights,
        step_prop_weights_alternative=jnp.zeros_like(init_prop_weights),
    )
    
    JTAP_data_init = JTAPInference(
        tracking=init_tracking_data,
        prediction=init_prediction_data,
        weight_data=init_weight_data,
        grid_data=init_grid_data,
        t=jnp.int32(0),
        resampled=jnp.bool(False),
        ESS=effective_sample_size(initial_weights),
        is_target_hidden=init_mos.is_target_hidden,
        is_target_partially_hidden=init_mos.is_target_partially_hidden,
        obs_is_fully_hidden=obs_is_fully_hidden,
        stopped_early=stopped_early
    )

    # Initial carry state
    initial_carry = (
        smc_key,
        init_mos,
        initial_weights,
        num_obj_pixels,
        init_prediction_data
    )
    
    # Perform scan
    _, (JTAP_data_steps, step_prop_retvals, xx) = jax.lax.scan(
        particle_filter_step_rg, 
        initial_carry, 
        jnp.arange(1,max_inference_steps)  # Start from 1
    )

    # Concatenate initial and step data
    all_JTAP_step_data = init_step_concat(JTAP_data_init, JTAP_data_steps)
    
    jtap_params = JTAPParams(
        max_prediction_steps=max_prediction_steps,
        max_inference_steps=max_inference_steps,
        num_particles=num_particles,
        ESS_threshold=ESS_threshold,
        simulate_every=mi.simulate_every,
        inference_input=mi
    )
    
    final_JTAP_data = JTAPMiceData(
        num_jtap_runs=None, # set outside jitted function
        inference=all_JTAP_step_data,
        params=jtap_params,
        step_prop_retvals=step_prop_retvals,
        init_prop_retval=init_prop_retval,
        stimulus = None, # set outside jitted function
        key_seed = None # set outside jitted function
    )

    return final_JTAP_data, xx

def run_jtap(key_seed, mi, ESS_proportion, stimulus, num_particles, max_inference_steps = None):
    # prep inputs
    if max_inference_steps is None:
        discrete_obs = stimulus.discrete_obs
        max_inference_steps = stimulus.num_frames
    else:
        discrete_obs = pad_obs_with_last_frame(stimulus.discrete_obs, max_inference_steps)
        
    initial_key = jax.random.key(key_seed)
    # cannot convert max inference steps and num particles to int32, as they are static args
    jax_jtap_data = run_jtap_(initial_key, mi, jnp.float32(ESS_proportion), jnp.asarray(discrete_obs), max_inference_steps, num_particles)
    if max_inference_steps is None:
        jtap_data = jtap_data_to_numpy(jax_jtap_data)
    else:
        jax_jtap_data = jax_jtap_data._replace(inference = multislice_pytree(jax_jtap_data.inference, np.arange(stimulus.num_frames)))
        jax_jtap_data = jax_jtap_data._replace(step_prop_retvals = multislice_pytree(jax_jtap_data.step_prop_retvals, np.arange(stimulus.num_frames - 1)))
        jax_jtap_data = jax_jtap_data._replace(params = jax_jtap_data.params._replace(max_inference_steps = jnp.int32(stimulus.num_frames)))
        jtap_data = jtap_data_to_numpy(jax_jtap_data)
    jtap_data = jtap_data._replace(stimulus = stimulus)
    jtap_data = jtap_data._replace(key_seed = key_seed)
    jtap_data = jtap_data._replace(num_jtap_runs = 1)
    return jtap_data

# NOTE need to figure out how to handle this, or keep it as it is
def run_parallel_jtap(num_jtap_runs, key_seed, mi, ESS_proportion, stimulus, num_particles, max_inference_steps = None):
    # prep inputs
    if max_inference_steps is None:
        discrete_obs = stimulus.discrete_obs
        max_inference_steps = stimulus.num_frames
    else:
        discrete_obs = pad_obs_with_last_frame(stimulus.discrete_obs, max_inference_steps)
        
    initial_key = jax.random.key(key_seed)
    # cannot convert max inference steps and num particles to int32, as they are static args
    initial_keys = jax.random.split(initial_key, num_jtap_runs)
    
    # Run JTAP
    jax_jtap_data, xx = jax.vmap(run_jtap_, in_axes = (0, None, None, None, None, None))(initial_keys, mi, jnp.float32(ESS_proportion), jnp.asarray(discrete_obs), max_inference_steps, num_particles)

    # Post-processing
    if max_inference_steps is None:
        jtap_data = jtap_data_to_numpy(jax_jtap_data)
    else:
        jax_jtap_data = jax_jtap_data._replace(inference = jtu.tree_map(lambda x: x[:, :stimulus.num_frames], jax_jtap_data.inference))
        jax_jtap_data = jax_jtap_data._replace(step_prop_retvals = jtu.tree_map(lambda x: x[:, :stimulus.num_frames - 1], jax_jtap_data.step_prop_retvals))
        jax_jtap_data = jax_jtap_data._replace(params = jax_jtap_data.params._replace(max_inference_steps = jnp.full(num_jtap_runs, stimulus.num_frames, dtype=jnp.int32)))
        jtap_data = jtap_data_to_numpy(jax_jtap_data)

    jtap_data = jtap_data._replace(stimulus = stimulus)
    jtap_data = jtap_data._replace(key_seed = key_seed)
    jtap_data = jtap_data._replace(num_jtap_runs = num_jtap_runs)
    
    return jtap_data, xx

# Jitted version of parallel JTAP
@partial(jax.jit, static_argnames=["num_jtap_runs", "max_inference_steps", "num_particles"])
def run_parallel_jtap_jitted_(initial_keys, mi, ESS_proportion, discrete_obs, num_jtap_runs, max_inference_steps, num_particles):
    return jax.vmap(run_jtap_, in_axes = (0, None, None, None, None, None))(initial_keys, mi, ESS_proportion, discrete_obs, max_inference_steps, num_particles)

import time
def run_parallel_jtap_jitted(num_jtap_runs, key_seed, mi, ESS_proportion, stimulus, num_particles):
    # prep inputs
    discrete_obs = stimulus.discrete_obs
    max_inference_steps = stimulus.num_frames
    initial_key = jax.random.key(key_seed)
    initial_keys = jax.random.split(initial_key, num_jtap_runs)
    start_time = time.time()
    jtap_data = jtap_data_to_numpy(run_parallel_jtap_jitted_(
        initial_keys, 
        mi, 
        jnp.float32(ESS_proportion), 
        jnp.asarray(discrete_obs), 
        num_jtap_runs, 
        max_inference_steps, 
        num_particles
    ))
    end_time = time.time()
    print(f"Time taken for parallel JTAP: {end_time - start_time} seconds")
    jtap_data = jtap_data._replace(stimulus = stimulus)
    jtap_data = jtap_data._replace(key_seed = key_seed)
    jtap_data = jtap_data._replace(num_jtap_runs = num_jtap_runs)
    return jtap_data