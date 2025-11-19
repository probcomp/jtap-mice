# %%
import jtap_mice
jtap_mice.set_jaxcache()
from jtap_mice.model import full_init_model, full_step_model, likelihood_model, stepper_model, get_render_args,is_ball_in_valid_position, red_green_sensor_readouts
from jtap_mice.inference import run_jtap, run_parallel_jtap, JTAPMiceData, pad_obs_with_last_frame
from jtap_mice.viz import rerun_jtap_stimulus, rerun_jtap_single_run, jtap_plot_rg_lines, red_green_viz_notebook, create_log_frequency_heatmaps
from jtap_mice.utils import load_red_green_stimulus, JTAPMiceStimulus, ChexModelInput, d2r, i_, f_, slice_pt, init_step_concat, discrete_obs_to_rgb, load_original_jtap_results, stack_pytrees, concat_pytrees, multislice_pytree
from jtap_mice.evaluation import JTAP_Decision_Model_Hyperparams, jtap_compute_beliefs, jtap_compute_decisions, jtap_compute_decision_metrics, JTAP_Metrics, JTAPMice_Beliefs, JTAP_Decisions, JTAP_Results
from jtap_mice.distributions import truncated_normal_sample, discrete_normal_sample
from jtap_mice.core import SuperPytree

import os
import copy
import time
import rerun as rr
import genjax
from genjax import gen, ChoiceMapBuilder as C
import jax
import jax.numpy as jnp
from jax.debug import print as jprint
import numpy as np
from tqdm import tqdm
import jax.tree_util as jtu
from functools import partial
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Tuple, NamedTuple


# Set seed for reproducible sampling
np.random.seed(42)
DECISION_MODEL_VERSION = "v4"
TUNING_NAME = "hypertuning_v6"

num_tuning_runs = 400
num_decision_runs_per_tuning_run = 30
particle_counts = [30] * num_tuning_runs

assert len(particle_counts) == num_tuning_runs

# Use empirical top 10% performer parameter means and SDs from report for uniform ranges:
# dir_val:       mean=1.463, sd=1.160          -> [0.303, 2.623]
# speed_val:     mean=0.044, sd=0.036          -> [0.008, 0.080]
# pos_sim_val:   mean=0.020, sd=0.016          -> [0.004, 0.036]
# pixel_corrupt: mean=0.212, sd=0.238          -> [0,    0.45]
# min_grid:      mean=0.267, sd=0.179          -> [0.088, 0.446]
# max_grid:      mean=1.304, sd=0.340          -> [0.964, 1.644]
# dir_mult:      mean=1.723, sd=1.439          -> [0.284, 3.162]
# speed_mult:    mean=1.808, sd=1.499          -> [0.309, 3.307]
# ESS_prop:      mean=0.366, sd=0.214          -> [0.152, 0.580]
# model_outlier: mean=0.021, sd=0.020          -> [0.001, 0.041]
# proposal_tau:  mean=70.644, sd=8.496         -> [62.148, 79.140]
# proposal_alpha:mean=6.345, sd=1.961          -> [4.384, 8.306]

dir_vals = np.random.uniform(0.303, 2.623, num_tuning_runs)
speed_vals = np.random.uniform(0.008, 0.080, num_tuning_runs)
pos_sim_vals = np.random.uniform(0.004, 0.036, num_tuning_runs)
pixel_corruption_probs = np.random.choice([0.01], size=num_tuning_runs)
min_grid_size_vals = np.random.uniform(0.088, 0.446, num_tuning_runs)
max_grid_size_vals = np.random.uniform(0.964, 1.644, num_tuning_runs)
dir_multipliers = np.random.uniform(0.284, 3.162, num_tuning_runs)
speed_multipliers = np.random.uniform(0.309, 3.307, num_tuning_runs)
ESS_props = np.random.uniform(0.152, 0.580, num_tuning_runs)
model_outlier_probs = np.random.uniform(0.001, 0.041, num_tuning_runs)
proposal_direction_outlier_taus = np.random.uniform(62.148, 79.140, num_tuning_runs)
proposal_direction_outlier_alphas = np.random.uniform(4.384, 8.306, num_tuning_runs)

# Decision model hyperparameters (Top 10% performers and SDs as uniform ranges)
# press_thresh_mean  [0.298, 0.614]
# press_thresh_std   [0.038, 0.230]
# tau_press_mean     [1.787, 6.793]
# tau_press_std      [0.382, 5.350]
# regular_delay_mean [1.442, 7.274]
# regular_delay_std  [0.744, 5.534]
# starting_delay_mean [3.607, 11.309]
# starting_delay_std  [0.711, 5.569]

# Expand to have for all decison runs (not only base tuning count)
press_thresh_means = np.random.uniform(0.298, 0.614, num_tuning_runs * num_decision_runs_per_tuning_run)
press_thresh_stds = np.random.uniform(0.038, 0.230, num_tuning_runs * num_decision_runs_per_tuning_run)

tau_press_means = np.random.uniform(1.787, 6.793, num_tuning_runs * num_decision_runs_per_tuning_run)
tau_press_stds = np.random.uniform(0.382, 5.350, num_tuning_runs * num_decision_runs_per_tuning_run)

regular_delay_means = np.random.uniform(1.442, 7.274, num_tuning_runs * num_decision_runs_per_tuning_run)
regular_delay_stds = np.random.uniform(0.744, 5.534, num_tuning_runs * num_decision_runs_per_tuning_run)

starting_delay_means = np.random.uniform(3.607, 11.309, num_tuning_runs * num_decision_runs_per_tuning_run)
starting_delay_stds = np.random.uniform(0.711, 5.569, num_tuning_runs * num_decision_runs_per_tuning_run)

os.makedirs(TUNING_NAME, exist_ok=True)

def run_all_cogsci_trials(num_jtap_runs, Model_Input, num_particles, smc_key_seed, ESS_proportion, tuning_idx = None, base_experiment_results = None):
    PIXEL_DENSITY = 10
    SKIP_T = 4
    jtap_stimuli = {}
    max_inference_steps = 0
    all_experiment_results = []
    experiment_results = copy.deepcopy(base_experiment_results)
    for i in tqdm(range(1,51), desc = "Loading all CogSci Stimuli"):
        COGSCI_TRIAL = f'E{i}'
        stimulus_path = f'/home/arijitdasgupta/jtap-mice/assets/stimuli/cogsci_2025_trials/{COGSCI_TRIAL}'
        jtap_stimulus = load_red_green_stimulus(stimulus_path, pixel_density = PIXEL_DENSITY, skip_t = SKIP_T)
        jtap_stimuli[COGSCI_TRIAL] = jtap_stimulus
        max_inference_steps = max(max_inference_steps, jtap_stimulus.num_frames)

    Model_Input.prepare_hyperparameters()

    for decision_idx in range(num_decision_runs_per_tuning_run):
        global_decision_idx = tuning_idx * num_decision_runs_per_tuning_run + decision_idx
        press_thresh_hyperparams = (press_thresh_means[global_decision_idx], press_thresh_stds[global_decision_idx], 0.0, 1.0)
        tau_press_hyperparams = (tau_press_means[global_decision_idx], tau_press_stds[global_decision_idx], np.arange(30))
        regular_delay_hyperparams = (regular_delay_means[global_decision_idx], regular_delay_stds[global_decision_idx], np.arange(30))
        starting_delay_hyperparams = (starting_delay_means[global_decision_idx], starting_delay_stds[global_decision_idx], np.arange(30))

        experiment_results['press_thresh_hyperparams'] = press_thresh_hyperparams
        experiment_results['tau_press_hyperparams'] = tau_press_hyperparams
        experiment_results['regular_delay_hyperparams'] = regular_delay_hyperparams
        experiment_results['starting_delay_hyperparams'] = starting_delay_hyperparams
        experiment_results['global_tuning_idx'] = global_decision_idx
        all_experiment_results.append(copy.deepcopy(experiment_results))

    for i, jtap_stimulus in enumerate(tqdm(jtap_stimuli, desc = "Running all CogSci Trials")):
        COGSCI_TRIAL = list(jtap_stimuli.keys())[i]
        jtap_stimulus = jtap_stimuli[COGSCI_TRIAL]
        Model_Input.prepare_scene_geometry(jtap_stimulus)
        smc_key_seed += 1
        JTAP_DATA, _ = run_parallel_jtap(num_jtap_runs, smc_key_seed, Model_Input, ESS_proportion, jtap_stimulus, num_particles, max_inference_steps = max_inference_steps)
        JTAPMice_Beliefs = jtap_compute_beliefs(JTAP_DATA)

        for decision_idx in range(num_decision_runs_per_tuning_run):
            press_thresh_hyperparams = all_experiment_results[decision_idx]['press_thresh_hyperparams']
            tau_press_hyperparams = all_experiment_results[decision_idx]['tau_press_hyperparams']
            regular_delay_hyperparams = all_experiment_results[decision_idx]['regular_delay_hyperparams']
            starting_delay_hyperparams = all_experiment_results[decision_idx]['starting_delay_hyperparams']

            jtap_decision_model_hyperparams = JTAP_Decision_Model_Hyperparams(
                key_seed = 123,
                pseudo_participant_multiplier = 50,
                press_thresh_hyperparams = press_thresh_hyperparams,
                tau_press_hyperparams = tau_press_hyperparams,
                hysteresis_delay_hyperparams = None,
                regular_delay_hyperparams = regular_delay_hyperparams,
                starting_delay_hyperparams = starting_delay_hyperparams
            )

            jtap_decisions, jtap_decision_model_params = jtap_compute_decisions(JTAPMice_Beliefs, jtap_decision_model_hyperparams, remove_keypress_to_save_memory = True, decision_model_version = DECISION_MODEL_VERSION)
            jtap_metrics = jtap_compute_decision_metrics(jtap_decisions, jtap_stimulus, partial_occlusion_in_targeted_analysis=True, ignore_uncertain_line=True)
            jtap_results = JTAP_Results(jtap_data = None, JTAPMice_Beliefs = None, jtap_decisions = jtap_decisions, jtap_metrics = jtap_metrics)
            all_experiment_results[decision_idx][COGSCI_TRIAL] = jtap_results

    return all_experiment_results

# %%
for tuning_idx in range(num_tuning_runs):

    # If you want to subsample runs, update logic here! Otherwise, runs all.
    if tuning_idx % 2 == 1:
        continue

    try:
        print(f"Running tuning run {tuning_idx+1} of {num_tuning_runs}")
        num_particles = particle_counts[tuning_idx]
        ESS_proportion = ESS_props[tuning_idx]
        dir_val = dir_vals[tuning_idx]
        pos_val = 10000.0 # hardcoded to a large number to avoid issues with position jumps
        speed_val = speed_vals[tuning_idx]
        pixel_corruption_prob = pixel_corruption_probs[tuning_idx]
        min_grid_size_val = min_grid_size_vals[tuning_idx]
        max_grid_size_val = max_grid_size_vals[tuning_idx]
        pos_sim_val = pos_sim_vals[tuning_idx]
        dir_multiplier = dir_multipliers[tuning_idx]
        speed_multiplier = speed_multipliers[tuning_idx]
        model_outlier_prob = model_outlier_probs[tuning_idx]
        proposal_direction_outlier_tau = proposal_direction_outlier_taus[tuning_idx]
        proposal_direction_outlier_alpha = proposal_direction_outlier_alphas[tuning_idx]
        print(f"Params: particles={num_particles} ESS_prop={ESS_proportion:.3f} dir_val={dir_val:.3f} speed_val={speed_val:.3f} pix_corrupt={pixel_corruption_prob:.3f} min_grid={min_grid_size_val:.3f} max_grid={max_grid_size_val:.3f} pos_sim={pos_sim_val:.3f} dir_mult={dir_multiplier:.3f} speed_mult={speed_multiplier:.3f} model_outlier={model_outlier_prob:.3f} prop_tau={proposal_direction_outlier_tau:.3f} prop_alpha={proposal_direction_outlier_alpha:.3f}")

        init_speed = speed_val
        init_direction = dir_val

        Model_Input = ChexModelInput(
            model_outlier_prob=f_(model_outlier_prob),
            proposal_direction_outlier_tau=d2r(proposal_direction_outlier_tau),
            proposal_direction_outlier_alpha=f_(proposal_direction_outlier_alpha),
            σ_speed_init_model=f_(init_speed),
            σ_direction_init_model=f_(init_direction),
            σ_pos=f_(pos_val),
            σ_speed=f_(speed_val),
            σ_NOCOL_direction=d2r(dir_val),
            σ_COL_direction=d2r(dir_val),
            pixel_corruption_prob=f_(pixel_corruption_prob),
            tile_size=i_(3),# can ignore
            σ_pixel_spatial=f_(1.0),# can ignore
            image_power_beta=f_(0.005),# can ignore
            max_speed=f_(1.0),# can ignore
            max_num_barriers=i_(10),# can ignore
            max_num_occ=i_(5),# can ignore
            num_x_grid=i_(6),
            num_y_grid=i_(6),
            grid_size_bounds=f_((min_grid_size_val, max_grid_size_val)),
            max_num_col_iters=f_(2), # can ignore
            simulate_every=i_(1), # can ignore
            σ_pos_sim=f_(pos_sim_val),
            σ_speed_sim=f_(speed_val),
            σ_NOCOL_direction_sim=d2r(dir_val),
            σ_COL_direction_sim=d2r(dir_val),
            σ_speed_occ=f_(0.0005),# can ignore
            σ_NOCOL_direction_occ=d2r(0.8),# can ignore
            σ_COL_direction_occ=d2r(0.8),# can ignore
            σ_pos_initprop=f_(0.02), # not consequential
            σ_speed_initprop=f_(speed_val*speed_multiplier),
            σ_speed_stepprop=f_(speed_val*speed_multiplier),
            σ_NOCOL_direction_initprop=d2r(dir_val*dir_multiplier),
            σ_NOCOL_direction_stepprop=d2r(dir_val*dir_multiplier),
            σ_COL_direction_prop=d2r(dir_val*dir_multiplier),
            σ_pos_stepprop=f_(0.01) # can ignore
        )

        smc_key_seed = np.random.randint(0, 1000000)

        base_experiment_results = {}
        base_experiment_results['smc_key_seed'] = smc_key_seed
        base_experiment_results['num_particles'] = num_particles
        base_experiment_results['ESS_proportion'] = ESS_proportion
        base_experiment_results['dir_val'] = dir_val
        base_experiment_results['speed_val'] = speed_val
        base_experiment_results['pixel_corruption_prob'] = pixel_corruption_prob
        base_experiment_results['min_grid_size_val'] = min_grid_size_val
        base_experiment_results['max_grid_size_val'] = max_grid_size_val
        base_experiment_results['pos_sim_val'] = pos_sim_val
        base_experiment_results['dir_multiplier'] = dir_multiplier
        base_experiment_results['speed_multiplier'] = speed_multiplier
        base_experiment_results['model_outlier_prob'] = model_outlier_prob
        base_experiment_results['proposal_direction_outlier_tau'] = proposal_direction_outlier_tau
        base_experiment_results['proposal_direction_outlier_alpha'] = proposal_direction_outlier_alpha

        all_experiment_results = run_all_cogsci_trials(
            num_jtap_runs = 50,
            Model_Input = Model_Input,
            num_particles = num_particles,
            smc_key_seed = smc_key_seed,
            ESS_proportion = ESS_proportion,
            tuning_idx = tuning_idx,
            base_experiment_results = base_experiment_results
        )

        for decision_idx in range(num_decision_runs_per_tuning_run):
            experiment_results = all_experiment_results[decision_idx]
            global_tuning_idx = experiment_results['global_tuning_idx']
            np.savez_compressed(f'{TUNING_NAME}/setup_{global_tuning_idx+1}.npz', experiment_results=experiment_results)

    except Exception as e:
        print(f"Error in tuning run {tuning_idx+1}: {e}")

