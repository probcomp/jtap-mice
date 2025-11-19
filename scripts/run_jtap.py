# Start of Selection
#!/usr/bin/env python3
"""
JTAP Runner Script

This script runs JTAP (Joint Tracking and Prediction) inference on red-green task stimuli.
It supports both single trial and batch processing modes.

Usage:
    # Single trial (default example: E50)
    uv run python run_jtap.py --stimulus_path assets/stimuli/cogsci_2025_trials/E50 --output_path jtap_output

    # Batch processing (all CogSci trials)
    uv run python run_jtap.py --stimulus_folder assets/stimuli/cogsci_2025_trials --output_path jtap_output --batch_mode

    # With custom parameters
    uv run python run_jtap.py --stimulus_path assets/stimuli/cogsci_2025_trials/E50 --num_jtap_runs 100 --num_particles 100 --skip_t 2
"""

import argparse
import os
import sys
import time
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track
from rich.text import Text

# Set up rich console, fallback to print if unavailable
if Console is not None:
    console = Console()
    def p(*args, **kwargs):
        console.print(*args, **kwargs)
else:
    console = None
    def p(*args, **kwargs):
        print(*args, **kwargs)

# Add the src directory to the path so we can import jtap_mice modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import jtap_mice
jtap_mice.set_jaxcache()

from jtap_mice.model import full_init_model, full_step_model, likelihood_model, stepper_model, get_render_args, is_ball_in_valid_position, red_green_sensor_readouts
from jtap_mice.inference import run_jtap, run_parallel_jtap, JTAPData
from jtap_mice.viz import jtap_plot_rg_lines, create_log_frequency_heatmaps
from jtap_mice.utils import load_red_green_stimulus, JTAPStimulus, ChexModelInput, d2r, i_, f_, slice_pt, init_step_concat, discrete_obs_to_rgb, load_original_jtap_results, stack_pytrees, concat_pytrees
from jtap_mice.evaluation import JTAP_Decision_Model_Hyperparams, jtap_compute_beliefs, jtap_compute_decisions, jtap_compute_decision_metrics, JTAP_Metrics, JTAP_Beliefs, JTAP_Decisions, JTAP_Results, compute_combined_metrics
from jtap_mice.distributions import truncated_normal_sample, discrete_normal_sample
from jtap_mice.core import SuperPytree

def load_hyperparams(config_path: str) -> Tuple[Dict, Dict]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_hyperparams = config['model_hyperparams']
    decision_hyperparams = config['decision_hyperparams']
    return model_hyperparams, decision_hyperparams

def format_hyperparam_value(val, float_digits=4):
    # For floats, show .3f or use scientific notation if needed; for arrays or lists, pretty-print; else str
    if isinstance(val, float):
        if abs(val) != 0 and (abs(val) < 0.001 or abs(val) > 1e4):
            return f"{val:.3e}"
        return f"{val:.{float_digits}f}"
    elif isinstance(val, (list, tuple)):
        parts = []
        for v in val:
            parts.append(format_hyperparam_value(v, float_digits=float_digits))
        return "[" + ", ".join(parts) + "]"
    else:
        return str(val)

def pretty_hyperparams_panel(title, hps_dict):
    # hps_dict: dict of hyperparams
    shown = {}
    for k, v in hps_dict.items():
        shown[k] = format_hyperparam_value(v)
    pretty_boxed_kv(title, shown)

def create_model_input(model_hyperparams: Dict) -> ChexModelInput:
    grid_size_bounds = f_(tuple(model_hyperparams['grid_size_bounds']))
    TILE_SIZE = 3
    SIGMA_PIXEL_SPATIAL = 1.0
    IMAGE_POWER_BETA = 0.005
    MAX_SPEED = 1.0
    MAX_NUM_BARRIERS = 10
    MAX_NUM_OCC = 10
    MAX_NUM_COL_ITERS = 2
    SIMULATE_EVERY = 1
    SIGMA_SPEED_OCC = 0.0005
    SIGMA_NOCOL_DIRECTION_OCC = 0.8
    SIGMA_COL_DIRECTION_OCC = 0.8
    SIGMA_POS_STEPPROP = 0.01

    return ChexModelInput(
        model_outlier_prob=f_(model_hyperparams.get('model_outlier_prob', 0.0)),
        proposal_direction_outlier_tau=d2r(model_hyperparams.get('proposal_direction_outlier_tau', 200.0)),
        proposal_direction_outlier_alpha=f_(model_hyperparams.get('proposal_direction_outlier_alpha', 50.0)),
        σ_speed_init_model=f_(model_hyperparams.get('σ_speed_init_model', model_hyperparams['σ_speed'])),
        σ_direction_init_model=d2r(model_hyperparams.get('σ_direction_init_model', model_hyperparams['σ_NOCOL_direction'])),
        σ_pos=f_(model_hyperparams['σ_pos']),
        σ_speed=f_(model_hyperparams['σ_speed']),
        σ_NOCOL_direction=d2r(model_hyperparams['σ_NOCOL_direction']),
        σ_COL_direction=d2r(model_hyperparams['σ_COL_direction']),
        pixel_corruption_prob=f_(model_hyperparams['pixel_corruption_prob']),
        tile_size=i_(TILE_SIZE),
        σ_pixel_spatial=f_(SIGMA_PIXEL_SPATIAL),
        image_power_beta=f_(IMAGE_POWER_BETA),
        max_speed=f_(MAX_SPEED),
        max_num_barriers=i_(MAX_NUM_BARRIERS),
        max_num_occ=i_(MAX_NUM_OCC),
        num_x_grid=i_(model_hyperparams['num_x_grid']),
        num_y_grid=i_(model_hyperparams['num_y_grid']),
        grid_size_bounds=grid_size_bounds,
        max_num_col_iters=f_(MAX_NUM_COL_ITERS),
        simulate_every=i_(SIMULATE_EVERY),
        σ_pos_sim=f_(model_hyperparams['σ_pos_sim']),
        σ_speed_sim=f_(model_hyperparams['σ_speed_sim']),
        σ_NOCOL_direction_sim=d2r(model_hyperparams['σ_NOCOL_direction_sim']),
        σ_COL_direction_sim=d2r(model_hyperparams['σ_COL_direction_sim']),
        σ_speed_occ=f_(SIGMA_SPEED_OCC),
        σ_NOCOL_direction_occ=d2r(SIGMA_NOCOL_DIRECTION_OCC),
        σ_COL_direction_occ=d2r(SIGMA_COL_DIRECTION_OCC),
        σ_pos_initprop=f_(model_hyperparams['σ_pos_initprop']),
        σ_speed_initprop=f_(model_hyperparams['σ_speed_initprop']),
        σ_speed_stepprop=f_(model_hyperparams['σ_speed_stepprop']),
        σ_NOCOL_direction_initprop=d2r(model_hyperparams['σ_NOCOL_direction_initprop']),
        σ_NOCOL_direction_stepprop=d2r(model_hyperparams['σ_NOCOL_direction_stepprop']),
        σ_COL_direction_prop=d2r(model_hyperparams['σ_COL_direction_prop']),
        σ_pos_stepprop=f_(SIGMA_POS_STEPPROP)
    )

def create_compressed_results(jtap_results: JTAP_Results) -> JTAP_Results:
    from jtap_mice.inference.jtap_types import WeightData, PredictionData, JTAPInference

    minimal_weight_data = WeightData(
        prop_weights=None,
        grid_weights=None,
        incremental_weights=None,
        prev_weights=None,
        final_weights=jtap_results.jtap_data.inference.weight_data.final_weights
    )

    minimal_prediction_data = PredictionData(
        x=None,
        y=None,
        speed=None,
        direction=None,
        collision_branch=None,
        rg=jtap_results.jtap_data.inference.prediction.rg
    )

    minimal_inference = JTAPInference(
        tracking=None,
        prediction=minimal_prediction_data,
        weight_data=minimal_weight_data,
        grid_data=None,
        t=jtap_results.jtap_data.inference.t,
        resampled=jtap_results.jtap_data.inference.resampled,
        ESS=jtap_results.jtap_data.inference.ESS,
        is_target_hidden=jtap_results.jtap_data.inference.is_target_hidden,
        is_target_partially_hidden=jtap_results.jtap_data.inference.is_target_partially_hidden,
        obs_is_fully_hidden=jtap_results.jtap_data.inference.obs_is_fully_hidden,
        stopped_early=jtap_results.jtap_data.inference.stopped_early
    )

    compressed_jtap_data = JTAPData(
        num_jtap_runs=jtap_results.jtap_data.num_jtap_runs,
        inference=minimal_inference,
        params=jtap_results.jtap_data.params,
        step_prop_retvals=None,
        init_prop_retval=None,
        key_seed=jtap_results.jtap_data.key_seed,
        stimulus=jtap_results.jtap_data.stimulus
    )

    return JTAP_Results(
        jtap_data=compressed_jtap_data,
        jtap_beliefs=jtap_results.jtap_beliefs,
        jtap_decisions=jtap_results.jtap_decisions,
        jtap_metrics=jtap_results.jtap_metrics
    )

def pretty_panel(title, value=None, subtitle=None, style="bold green", highlight=True):
    if console is not None:
        panel_text = Text(str(value) if value is not None else "")
        # Strip all [italic] and [blue] and similar, by using only the raw value string.
        panel = Panel(panel_text, title=str(title), subtitle=str(subtitle) if subtitle else None, style="green", highlight=highlight)
        console.print(panel)
        return panel
    else:
        return f"[{title}] {value}" if value is not None else f"[{title}]"

def pretty_metrics(metrics, title="METRICS SUMMARY"):
    if console is None or Table is None:
        print(title)
        print(f"Model RMSE: {metrics.model_metrics.rmse_loss:.4f}")
        print(f"Model Occlusion RMSE: {metrics.model_metrics.occ_rmse_loss:.4f}")
        print(f"Frozen RMSE: {metrics.frozen_metrics.rmse_loss:.4f}")
        print(f"Frozen Occlusion RMSE: {metrics.frozen_metrics.occ_rmse_loss:.4f}")
        print(f"Decaying RMSE: {metrics.decaying_metrics.rmse_loss:.4f}")
        print(f"Decaying Occlusion RMSE: {metrics.decaying_metrics.occ_rmse_loss:.4f}")
    else:
        table = Table(title=title, box=box.SIMPLE_HEAD)
        table.add_column("Metric", justify="right")
        table.add_column("Value", justify="right", style="bold cyan")
        table.add_row("Model RMSE", f"{metrics.model_metrics.rmse_loss:.4f}")
        table.add_row("Model Occlusion RMSE", f"{metrics.model_metrics.occ_rmse_loss:.4f}")
        table.add_row("Frozen RMSE", f"{metrics.frozen_metrics.rmse_loss:.4f}")
        table.add_row("Frozen Occlusion RMSE", f"{metrics.frozen_metrics.occ_rmse_loss:.4f}")
        table.add_row("Decaying RMSE", f"{metrics.decaying_metrics.rmse_loss:.4f}")
        table.add_row("Decaying Occlusion RMSE", f"{metrics.decaying_metrics.occ_rmse_loss:.4f}")
        console.print(table)

def pretty_rule(title, style="bold magenta"):
    if console is not None:
        # Remove any explicit color/italic markup from title.
        console.rule(Text(str(title)))
    else:
        print("=" * 10 + f" {title} " + "=" * 10)

def pretty_warning(msg):
    if console is not None:
        console.print(f"[bold yellow]⚠️  {msg}[/bold yellow]")
    else:
        print("WARNING:", msg)

def pretty_error(msg):
    if console is not None:
        console.print(f"[bold red]❌ {msg}[/bold red]")
    else:
        print("ERROR:", msg)

def pretty_info(msg, style="bold blue"):
    if console is not None:
        # Strip rich formatting codes from message automatically, but retain any color by default Panel
        if isinstance(msg, str):
            # Remove [italic] or [blue] tags if they made it in
            msg_stripped = msg.replace("[italic]", "").replace("[/italic]", "").replace("[blue]", "").replace("[/blue]", "")
            console.print(Text(msg_stripped))
        else:
            console.print(msg)
    else:
        print(msg)

def pretty_success(msg):
    if console is not None:
        if isinstance(msg, Panel):
            console.print(msg)
        else:
            # Remove any [italic] tags, etc.
            msg_stripped = str(msg).replace("[italic]", "").replace("[/italic]", "")
            console.print(f"[bold green]✅ {msg_stripped}[/bold green]")
    else:
        print(msg)

def pretty_boxed_kv(title, kv: dict, title_style="bold", key_style="cyan", val_style="magenta"):
    if console is not None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left", style=key_style)
        grid.add_column(justify="left", style=val_style)
        for k, v in kv.items():
            grid.add_row(str(k), str(v))
        panel = Panel(grid, title=str(title), title_align="left", border_style="green")
        console.print(panel)
    else:
        print(f"--- {title} ---")
        for k, v in kv.items():
            print(f"{k}: {v}")

def create_decision_hyperparams(decision_hyperparams: Dict) -> Tuple[JTAP_Decision_Model_Hyperparams, str]:
    delay_range = np.arange(30)
    hysteresis_delay_hyperparams = None
    if decision_hyperparams.get('hysteresis_delay_hyperparams') is not None:
        hysteresis_delay_hyperparams = (
            decision_hyperparams['hysteresis_delay_hyperparams']['mean'],
            decision_hyperparams['hysteresis_delay_hyperparams']['std'],
            delay_range
        )

    decision_model_version = decision_hyperparams.get('decision_model_version', 'v4')

    return JTAP_Decision_Model_Hyperparams(
        key_seed=decision_hyperparams['key_seed'],
        pseudo_participant_multiplier=decision_hyperparams['pseudo_participant_multiplier'],
        press_thresh_hyperparams=(
            decision_hyperparams['press_thresh_hyperparams']['mean'],
            decision_hyperparams['press_thresh_hyperparams']['std'],
            decision_hyperparams['press_thresh_hyperparams']['min'],
            decision_hyperparams['press_thresh_hyperparams']['max']
        ),
        tau_press_hyperparams=(
            decision_hyperparams['tau_press_hyperparams']['mean'],
            decision_hyperparams['tau_press_hyperparams']['std'],
            delay_range
        ),
        hysteresis_delay_hyperparams=hysteresis_delay_hyperparams,
        regular_delay_hyperparams=(
            decision_hyperparams['regular_delay_hyperparams']['mean'],
            decision_hyperparams['regular_delay_hyperparams']['std'],
            delay_range
        ),
        starting_delay_hyperparams=(
            decision_hyperparams['starting_delay_hyperparams']['mean'],
            decision_hyperparams['starting_delay_hyperparams']['std'],
            delay_range
        )
    ), decision_model_version

def run_single_trial(
    stimulus_path: str,
    model_input: ChexModelInput,
    decision_hyperparams: JTAP_Decision_Model_Hyperparams,
    decision_model_version: str,
    num_jtap_runs: int,
    num_particles: int,
    ess_proportion: float,
    pixel_density: int,
    skip_t: int,
    smc_key_seed: int,
    output_dir: Path,
    save_compressed: bool = False,
) -> JTAP_Results:
    pretty_rule("SINGLE TRIAL")
    pretty_info(f"Loading stimulus from:\n  {stimulus_path}")
    jtap_stimulus = load_red_green_stimulus(stimulus_path, pixel_density=pixel_density, skip_t=skip_t)

    model_input.prepare_hyperparameters()
    model_input.prepare_scene_geometry(jtap_stimulus)

    p("")
    pretty_info(f"Running JTAP inference: {num_jtap_runs} runs × {num_particles} particles/run ...")
    start_time = time.time()
    max_inference_steps = jtap_stimulus.num_frames
    jtap_data, _ = run_parallel_jtap(
        num_jtap_runs, smc_key_seed, model_input, ess_proportion, jtap_stimulus, num_particles, max_inference_steps=max_inference_steps
    )
    end_time = time.time()
    pretty_success(f"JTAP inference completed in {end_time - start_time:.2f} seconds (includes possible JAX JIT compilation)")

    pretty_info("Computing beliefs...")
    jtap_beliefs = jtap_compute_beliefs(jtap_data)

    pretty_info("Computing decisions...")
    jtap_decisions, jtap_decision_model_params = jtap_compute_decisions(
        jtap_beliefs, decision_hyperparams, remove_keypress_to_save_memory=True, decision_model_version=decision_model_version
    )

    if jtap_stimulus.human_data is not None:
        pretty_info("Computing metrics...")
        jtap_metrics = jtap_compute_decision_metrics(
            jtap_decisions, jtap_stimulus, partial_occlusion_in_targeted_analysis=True, ignore_uncertain_line=True
        )
    else:
        pretty_warning("No human data available. Skipping metrics computation.")
        jtap_metrics = None

    jtap_results = JTAP_Results(
        jtap_data=jtap_data,
        jtap_beliefs=jtap_beliefs,
        jtap_decisions=jtap_decisions,
        jtap_metrics=jtap_metrics,
    )

    trial_name = Path(stimulus_path).name
    trial_output_dir = output_dir / trial_name
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    if save_compressed:
        compressed_results = create_compressed_results(jtap_results)
        results_path = trial_output_dir / f"{trial_name}_results_compressed.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(compressed_results, f)
        pretty_success(Panel(f"Compressed results saved to:\n{results_path}", title="Saved"))
    else:
        results_path = trial_output_dir / f"{trial_name}_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(jtap_results, f)
        pretty_success(Panel(f"Uncompressed results saved to:\n{results_path}", title="Saved"))

    pretty_info("Creating plots...")

    include_human = jtap_stimulus.human_data is not None

    fig = jtap_plot_rg_lines(
        jtap_decisions,
        stimulus=jtap_stimulus,
        show="model",
        include_baselines=True,
        include_human=include_human,
        jtap_metrics=jtap_metrics,
        return_fig=True,
        include_stimulus=True,
    )
    plot_path = trial_output_dir / f"{trial_name}_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pretty_success(Panel(f"Main plot saved to:\n{plot_path}", title="Plot"))

    # ==== RAW BELIEFS PLOT ====
    pretty_info("Creating raw beliefs plot...")
    raw_beliefs_fig = jtap_plot_rg_lines(
        jtap_beliefs,
        stimulus=jtap_stimulus,
        show="model",
        include_baselines=True,
        include_human=include_human,
        remove_legend=True,
        show_std_band=True,
        jtap_run_idx=None,
        include_start_frame=True,
        show_all_beliefs=False,
        plot_stat="median",
        include_stimulus=True,
        return_fig=True
    )
    raw_beliefs_plot_path = trial_output_dir / f"{trial_name}_raw_beliefs.png"
    raw_beliefs_fig.savefig(raw_beliefs_plot_path, dpi=300, bbox_inches="tight")
    plt.close(raw_beliefs_fig)
    pretty_success(Panel(f"Raw beliefs plot saved to:\n{raw_beliefs_plot_path}", title="Plot"))

    if jtap_metrics is not None:
        pretty_metrics(jtap_metrics, title=f"Trial Metrics Summary • {trial_name}")

    return jtap_results

def run_batch_trials(
    stimulus_folder: str,
    model_input: ChexModelInput,
    decision_hyperparams: JTAP_Decision_Model_Hyperparams,
    decision_model_version: str,
    num_jtap_runs: int,
    num_particles: int,
    ess_proportion: float,
    pixel_density: int,
    skip_t: int,
    smc_key_seed: int,
    output_dir: Path,
    save_compressed: bool = False,
) -> Dict[str, JTAP_Results]:
    pretty_rule("BATCH PROCESSING")
    stimulus_path = Path(stimulus_folder)
    if not stimulus_path.exists():
        raise ValueError(f"Stimulus folder does not exist: {stimulus_folder}")

    if save_compressed:
        pretty_info("Saving compressed results...")
    else:
        pretty_info("Saving uncompressed results...")

    trial_folders = [f for f in stimulus_path.iterdir() if f.is_dir()]
    trial_folders.sort()
    if not trial_folders:
        pretty_error(f"No trial folders found in: {stimulus_folder}")
        raise ValueError(f"No trial folders found in: {stimulus_folder}")
    pretty_info(f"Found {len(trial_folders)} trials to process.")

    pretty_info("Loading all stimuli...")

    jtap_stimuli = {}
    max_inference_steps = 0

    # rich progress bar if possible
    if console is not None:
        trial_folder_iter = track(trial_folders, description="Loading stimuli", transient=True)
    else:
        trial_folder_iter = tqdm(trial_folders, desc="Loading stimuli")

    for trial_folder in trial_folder_iter:
        trial_name = trial_folder.name
        try:
            jtap_stimulus = load_red_green_stimulus(str(trial_folder), pixel_density=pixel_density, skip_t=skip_t)
            jtap_stimuli[trial_name] = jtap_stimulus
            max_inference_steps = max(max_inference_steps, jtap_stimulus.num_frames)
        except Exception as e:
            pretty_warning(f"Failed to load stimulus {trial_name}: {e}")
            continue

    # pretty infor to tell max inference steps
    pretty_info(f"Maximum inference steps per trial: {max_inference_steps}")

    if not jtap_stimuli:
        raise ValueError("No stimuli could be loaded successfully")

    pretty_success(Panel(f"Successfully loaded {len(jtap_stimuli)} stimuli!", title="Input Data"))

    stimulus_folder_path = Path(stimulus_folder)
    batch_folder_name = stimulus_folder_path.name
    batch_output_dir = output_dir / batch_folder_name
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    pretty_info(f"Outputting results to {batch_output_dir}")

    model_input.prepare_hyperparameters()
    experiment_results = {}
    current_smc_key_seed = smc_key_seed

    # rich progress bar if possible
    if console is not None:
        trial_iter = track(list(jtap_stimuli.items()), description="Processing trials", transient=True)
    else:
        trial_iter = tqdm(jtap_stimuli.items(), desc="Processing trials")

    for trial_name, jtap_stimulus in trial_iter:
        try:
            model_input.prepare_scene_geometry(jtap_stimulus)
            start_time = time.time()
            jtap_data, _ = run_parallel_jtap(
                num_jtap_runs, current_smc_key_seed, model_input, ess_proportion, jtap_stimulus, num_particles, max_inference_steps=max_inference_steps
            )
            end_time = time.time()

            jtap_beliefs = jtap_compute_beliefs(jtap_data)
            jtap_decisions, _ = jtap_compute_decisions(
                jtap_beliefs, decision_hyperparams, remove_keypress_to_save_memory=True, decision_model_version=decision_model_version
            )
            if jtap_stimulus.human_data is not None:
                jtap_metrics = jtap_compute_decision_metrics(
                    jtap_decisions, jtap_stimulus, partial_occlusion_in_targeted_analysis=True, ignore_uncertain_line=True
                )
            else:
                jtap_metrics = None

            jtap_results = JTAP_Results(
                jtap_data=jtap_data,
                jtap_beliefs=jtap_beliefs,
                jtap_decisions=jtap_decisions,
                jtap_metrics=jtap_metrics,
            )
            experiment_results[trial_name] = jtap_results

            trial_output_dir = batch_output_dir / trial_name
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            if save_compressed:
                compressed_results = create_compressed_results(jtap_results)
                results_path = trial_output_dir / f"{trial_name}_results_compressed.pkl"
                with open(results_path, "wb") as f:
                    pickle.dump(compressed_results, f)
            else:
                results_path = trial_output_dir / f"{trial_name}_results.pkl"
                with open(results_path, "wb") as f:
                    pickle.dump(jtap_results, f)

            include_human = jtap_stimulus.human_data is not None

            fig = jtap_plot_rg_lines(
                jtap_decisions,
                stimulus=jtap_stimulus,
                show="model",
                include_baselines=True,
                include_human=include_human,
                jtap_metrics=jtap_metrics,
                return_fig=True,
                include_stimulus=True,
            )
            plot_path = trial_output_dir / f"{trial_name}_plot.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Also plot and save raw beliefs plot
            raw_beliefs_fig = jtap_plot_rg_lines(
                jtap_beliefs,
                stimulus=jtap_stimulus,
                show="model",
                include_human=include_human,
                include_baselines=True,
                remove_legend=True,
                show_std_band=True,
                jtap_run_idx=None,
                include_start_frame=True,
                show_all_beliefs=False,
                plot_stat="median",
                include_stimulus=True,
                return_fig=True
            )
            raw_beliefs_plot_path = trial_output_dir / f"{trial_name}_raw_beliefs.png"
            raw_beliefs_fig.savefig(raw_beliefs_plot_path, dpi=300, bbox_inches="tight")
            plt.close(raw_beliefs_fig)

            current_smc_key_seed += 1

        except Exception as e:
            pretty_error(f"Error processing trial {trial_name}: {e}")
            continue

    if experiment_results:
        pretty_info("\nSaving combined results...")
        if save_compressed:
            compressed_experiment_results = {}
            for trial_name, jtap_results in experiment_results.items():
                compressed_experiment_results[trial_name] = create_compressed_results(jtap_results)
            combined_results_path = batch_output_dir / "all_trials_results_compressed.pkl"
            with open(combined_results_path, "wb") as f:
                pickle.dump(compressed_experiment_results, f)
            pretty_success(Panel(f"Compressed combined results saved to:\n{combined_results_path}", title="Saved"))
        else:
            combined_results_path = batch_output_dir / "all_trials_results.pkl"
            with open(combined_results_path, "wb") as f:
                pickle.dump(experiment_results, f)
            pretty_success(Panel(f"Combined results saved to:\n{combined_results_path}", title="Saved"))

        pretty_info("Creating log frequency heatmaps...")
        try:
            trials_with_human_data = {
                name: results
                for name, results in experiment_results.items()
                if jtap_stimuli[name].human_data is not None
            }
            if trials_with_human_data:
                stimuli_with_human_data = {
                    name: stimulus for name, stimulus in jtap_stimuli.items() if name in trials_with_human_data
                }

                combined_metrics = compute_combined_metrics(
                    trial_results=trials_with_human_data,
                    jtap_stimuli=stimuli_with_human_data,
                    ignore_uncertain_line=True,
                    include_partial_occlusion=True,
                )

                fig = create_log_frequency_heatmaps(
                    combined_metrics,
                    targeted_analysis=True,
                    weighted=True,
                    cmap="Oranges",
                    cmap_reverse=False,
                    model_name="JTAP",
                    bins=25,
                )
                heatmap_path = batch_output_dir / "log_frequency_heatmaps.png"
                fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                pretty_success(Panel(f"Log frequency heatmaps saved to:\n{heatmap_path}", title="Visualization"))

                pretty_rule("BATCH PROCESSING METRICS SUMMARY")
                pretty_boxed_kv(
                    "Trial counts",
                    {
                        "Total trials processed": len(experiment_results),
                        "Trials with human data": len(trials_with_human_data),
                        "Trials without human data": len(experiment_results) - len(trials_with_human_data)
                    }
                )
                pretty_metrics(combined_metrics, title="Combined Metrics (All Trials With Human Data)")
            else:
                pretty_warning("No trials with human data available. Skipping combined metrics and heatmaps.")
        except Exception as e:
            pretty_warning(f"Failed to create log frequency heatmaps: {e}")

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run JTAP inference on red-green task stimuli")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stimulus_path", type=str, help="Path to single stimulus folder (e.g., assets/stimuli/cogsci_2025_trials/E50)")
    group.add_argument("--stimulus_folder", type=str, help="Path to folder containing multiple stimulus folders (e.g., assets/stimuli/cogsci_2025_trials)")

    parser.add_argument("--output_path", type=str, default="jtap_output", help="Output directory path (default: jtap_output)")
    parser.add_argument("--config", type=str, default="configs/jtap_hyperparams.yaml", help="Path to hyperparameters config file (default: configs/jtap_hyperparams.yaml)")
    parser.add_argument("--num_jtap_runs", type=int, default=50, help="Number of parallel runs (default: 50)")
    parser.add_argument("--num_particles", type=int, default=50, help="Number of particles for SMC (default: 50)")
    parser.add_argument("--ess_proportion", type=float, default=0.09, help="ESS proportion for resampling (default: 0.09)")
    parser.add_argument("--pixel_density", type=int, default=10, help="Pixel density for stimulus loading (default: 10)")
    parser.add_argument("--skip_t", type=int, default=4, help="Skip frames parameter (default: 4)")
    parser.add_argument("--smc_key_seed", type=int, default=None, help="Random seed for SMC (default: random)")
    parser.add_argument("--batch_mode", action="store_true", help="Enable batch processing mode (required when using --stimulus_folder)")
    parser.add_argument("--save_compressed", action="store_true", help="Save compressed data (removes keypress data and compresses JTAP_DATA to minimum needed for belief recomputation)")

    args = parser.parse_args()

    if args.stimulus_folder and not args.batch_mode:
        parser.error("--batch_mode is required when using --stimulus_folder")
    if args.stimulus_path and args.batch_mode:
        parser.error("--batch_mode cannot be used with --stimulus_path")

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    output_dir = project_root / args.output_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smc_key_seed is None:
        args.smc_key_seed = np.random.randint(0, 1000000)

    # --- JTAP RUN configuration panel ---
    pretty_rule("JTAP RUN", style="bold green")
    pretty_boxed_kv("Run configuration", {
        "SMC key seed": args.smc_key_seed,
        "Output directory": output_dir,
        "Config file": config_path,
        "Particles / run": args.num_particles,
        "JTAP runs": args.num_jtap_runs,
        "ESS proportion": args.ess_proportion,
        "Pixel density": args.pixel_density,
        "Skip frames": args.skip_t
    })

    pretty_info(f"Loading hyperparameters from: {config_path}")
    # --- Load hyperparams ---
    model_hyperparams, decision_hyperparams = load_hyperparams(str(config_path))

    # --- Show hyperparameters, as requested ---
    # (1) JTAP BAYESIAN INFERENCE hyperparameters
    pretty_rule("JTAP BAYESIAN INFERENCE HYPERPARAMETERS", style="bold cyan")
    pretty_hyperparams_panel("JTAP Inference Model Hyperparams", model_hyperparams)

    # (2) JTAP DECISION MODEL hyperparameters
    pretty_rule("JTAP DECISION MODEL HYPERPARAMETERS", style="bold magenta")
    pretty_hyperparams_panel("JTAP Decision Model Hyperparams", decision_hyperparams)

    model_input = create_model_input(model_hyperparams)
    decision_hyperparams_obj, decision_model_version = create_decision_hyperparams(decision_hyperparams)

    if args.stimulus_path:
        stimulus_path = str(project_root / args.stimulus_path)
        if not Path(stimulus_path).exists():
            pretty_error(f"Stimulus path not found: {stimulus_path}")
            raise FileNotFoundError(f"Stimulus path not found: {stimulus_path}")

        pretty_rule(f"Running single trial: {args.stimulus_path}", style="bold blue")
        results = run_single_trial(
            stimulus_path=stimulus_path,
            model_input=model_input,
            decision_hyperparams=decision_hyperparams_obj,
            decision_model_version=decision_model_version,
            num_jtap_runs=args.num_jtap_runs,
            num_particles=args.num_particles,
            ess_proportion=args.ess_proportion,
            pixel_density=args.pixel_density,
            skip_t=args.skip_t,
            smc_key_seed=args.smc_key_seed,
            output_dir=output_dir,
            save_compressed=args.save_compressed,
        )
        pretty_success("Single trial completed successfully!")

    else:
        stimulus_folder = str(project_root / args.stimulus_folder)
        if not Path(stimulus_folder).exists():
            pretty_error(f"Stimulus folder not found: {stimulus_folder}")
            raise FileNotFoundError(f"Stimulus folder not found: {stimulus_folder}")

        pretty_rule(f"Running batch trials: {args.stimulus_folder}", style="bold blue")
        results = run_batch_trials(
            stimulus_folder=stimulus_folder,
            model_input=model_input,
            decision_hyperparams=decision_hyperparams_obj,
            decision_model_version=decision_model_version,
            num_jtap_runs=args.num_jtap_runs,
            num_particles=args.num_particles,
            ess_proportion=args.ess_proportion,
            pixel_density=args.pixel_density,
            skip_t=args.skip_t,
            smc_key_seed=args.smc_key_seed,
            output_dir=output_dir,
            save_compressed=args.save_compressed,
        )
        pretty_success(f"Batch processing completed! Processed [bold]{len(results)}[/bold] trials.")

if __name__ == "__main__":
    main()