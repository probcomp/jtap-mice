#!/usr/bin/env python3
"""
Run JTAP on all trials of a single stimulus JSON and save one LR-belief plot PNG per trial.
Config is loaded from a YAML file; CLI flags and Python kwargs override YAML.

Time axis: The plot x-axis (seconds) uses stimulus.fps and stimulus.skip_t from the JSON
config (FRAMES_PER_SECOND) and the loaded stimulus. duration_sec = num_frames * skip_t / fps.
So a trial with 21 frames at 20 fps has duration 1.05 s; short trials (e.g. MIN_TRIAL_SECONDS=1)
are valid. Use --verbose to log num_frames, fps, and duration per trial.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Agg backend before any other matplotlib import for fast headless saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import yaml
from tqdm import tqdm

# Ensure package is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import jtap_mice
from jtap_mice.inference import run_parallel_jtap
from jtap_mice.viz import jtap_plot_lr_lines
from jtap_mice.utils import (
    load_left_right_stimulus,
    get_trial_keys_and_max_frames,
    get_assets_dir,
    ChexModelInput,
    d2r,
)
from jtap_mice.evaluation import jtap_compute_beliefs


DEFAULT_CONFIG_PATH = _SCRIPT_DIR / "jtap_run_config.yaml"


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_occlusion_regions(s: str | None):
    """Parse occlusion_regions from string like '4.1,3.2' or '4.1,3.2 5.7,3.2' -> list of [x_left, length]."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    if isinstance(s, list):
        return s if len(s) > 0 else None
    out = []
    for part in s.strip().split():
        part = part.strip()
        if not part:
            continue
        xs = part.split(",")
        if len(xs) != 2:
            raise ValueError(f"Each occlusion region must be 'x_left,length', got {part!r}")
        out.append([float(xs[0].strip()), float(xs[1].strip())])
    return out if out else None


def _build_model_input_from_config(cfg: dict):
    """Build ChexModelInput from config dict (YAML keys: sigma_*, proposal_direction_outlier_tau_deg, etc.)."""
    tau_rad = d2r(float(cfg.get("proposal_direction_outlier_tau_deg", 40.0)))
    return ChexModelInput(
        model_outlier_prob=float(cfg.get("model_outlier_prob", 0.0)),
        proposal_direction_outlier_tau=float(tau_rad),
        proposal_direction_outlier_alpha=float(cfg.get("proposal_direction_outlier_alpha", 3.5)),
        σ_pos_model=float(cfg.get("sigma_pos_model", 0.5)),
        σ_speed_model=float(cfg.get("sigma_speed_model", 0.075)),
        model_direction_flip_prob=float(cfg.get("model_direction_flip_prob", 0.025)),
        pixel_corruption_prob=float(cfg.get("pixel_corruption_prob", 0.01)),
        tile_size=int(cfg.get("tile_size", 3)),
        σ_pixel_spatial=float(cfg.get("sigma_pixel_spatial", 1.0)),
        image_power_beta=float(cfg.get("image_power_beta", 0.005)),
        max_speed=float(cfg.get("max_speed", 1.0)),
        max_num_occ=int(cfg.get("max_num_occ", 5)),
        num_x_grid=int(cfg.get("num_x_grid", 8)),
        grid_size_bounds=tuple(cfg.get("grid_size_bounds", [0.2, 1.2])),
        simulate_every=int(cfg.get("simulate_every", 1)),
        σ_pos_simulation=float(cfg.get("sigma_pos_simulation", 0.05)),
        σ_speed_simulation=float(cfg.get("sigma_speed_simulation", 0.075)),
        simulation_direction_flip_prob=float(cfg.get("simulation_direction_flip_prob", 0.005)),
        σ_pos_initprop=float(cfg.get("sigma_pos_initprop", 0.02)),
        proposal_direction_flip_prob=float(cfg.get("proposal_direction_flip_prob", 0.05)),
        σ_speed_stepprop=float(cfg.get("sigma_speed_stepprop", 0.01)),
        σ_pos_stepprop=float(cfg.get("sigma_pos_stepprop", 0.01)),
    )


def _load_trial_list_file(path: str | Path) -> list[str]:
    """Load trial keys from a file: one key per line, strip whitespace, skip blank lines and # comments."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trial list file not found: {path}")
    keys = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keys.append(line)
    return keys


def _merge_options(config: dict, args: argparse.Namespace, kwargs: dict) -> dict:
    """Merge config, parsed args, and kwargs. Later overrides earlier."""
    opts = dict(config)
    # Args
    if getattr(args, "stimulus_path", None) is not None:
        opts["stimulus_path"] = args.stimulus_path
    if getattr(args, "output_dir", None) is not None:
        opts["output_dir"] = args.output_dir
    if getattr(args, "config", None) is not None:
        pass  # already loaded
    for key in ("pixel_density", "skip_t", "num_particles", "ESS_proportion", "num_jtap_runs", "key_seed",
                "occlusion_regions", "save_dpi", "dry_run", "limit_trials", "verbose", "trial_list"):
        val = getattr(args, key, None)
        if val is not None:
            opts[key] = val
    # Kwargs override
    for k, v in kwargs.items():
        if v is not None:
            opts[k] = v
    return opts


def run(
    stimulus_path: str | Path,
    output_dir: str | Path = "jtap_plots",
    config_path: str | Path | None = None,
    pixel_density: int | None = None,
    skip_t: int | None = None,
    num_particles: int | None = None,
    ESS_proportion: float | None = None,
    num_jtap_runs: int | None = None,
    key_seed: int | None = None,
    occlusion_regions: list | str | None = None,
    save_dpi: int | None = None,
    dry_run: bool = False,
    limit_trials: int | None = None,
    verbose: bool = False,
    trial_list: str | Path | None = None,
    **kwargs,
) -> None:
    """
    Run JTAP on all trials of a single JSON and save one PNG per trial.
    All parameters can be overridden; defaults come from config YAML.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_config(config_path)

    # Pass through all run() params so CLI/kwargs override config (no param should be None here for overrides)
    args = argparse.Namespace(
        stimulus_path=str(stimulus_path),
        output_dir=output_dir,
        config=None,
        pixel_density=pixel_density,
        skip_t=skip_t,
        num_particles=num_particles,
        ESS_proportion=ESS_proportion,
        num_jtap_runs=num_jtap_runs,
        key_seed=key_seed,
        occlusion_regions=occlusion_regions,
        save_dpi=save_dpi,
        dry_run=dry_run,
        limit_trials=limit_trials,
        verbose=verbose,
        trial_list=trial_list,
    )
    opts = _merge_options(config, args, kwargs)

    stimulus_path = Path(opts["stimulus_path"])
    if not stimulus_path.is_file():
        raise FileNotFoundError(f"Stimulus JSON not found: {stimulus_path}")
    output_dir = Path(opts["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_density = int(opts["pixel_density"])
    skip_t = int(opts["skip_t"])
    num_particles = int(opts["num_particles"])
    ESS_proportion = float(opts["ESS_proportion"])
    num_jtap_runs = int(opts["num_jtap_runs"])
    key_seed = opts["key_seed"]
    if key_seed is None:
        key_seed = int(np.random.randint(0, 2**31))
    else:
        key_seed = int(key_seed)
    occlusion_regions = opts.get("occlusion_regions")
    if isinstance(occlusion_regions, str):
        occlusion_regions = _parse_occlusion_regions(occlusion_regions)
    elif occlusion_regions is not None and not isinstance(occlusion_regions, list):
        occlusion_regions = None
    save_dpi = int(opts.get("save_dpi", 100))
    dry_run = bool(opts.get("dry_run", False))
    limit_trials = opts.get("limit_trials")
    verbose = bool(opts.get("verbose", False))
    trial_list_path = opts.get("trial_list")

    jtap_mice.set_jaxcache()
    model_input = _build_model_input_from_config(opts)
    model_input.prepare_hyperparameters()

    trial_keys, max_inference_steps = get_trial_keys_and_max_frames(str(stimulus_path), skip_t)
    if trial_list_path is not None:
        file_keys = _load_trial_list_file(trial_list_path)
        valid = set(trial_keys)
        trial_keys = [k for k in file_keys if k in valid]
        missing = [k for k in file_keys if k not in valid]
        if missing:
            print(f"Trial list: {len(missing)} key(s) not in JSON (skipped): {missing[:20]}{'...' if len(missing) > 20 else ''}", file=sys.stderr)
        if not trial_keys:
            print("No trials to run after applying trial list.", file=sys.stderr)
            return
    if limit_trials is not None:
        trial_keys = trial_keys[: int(limit_trials)]
    json_stem = stimulus_path.stem

    if dry_run:
        print(f"Stimulus: {stimulus_path}")
        print(f"Trial keys ({len(trial_keys)}): {trial_keys[:10]}{'...' if len(trial_keys) > 10 else ''}")
        print(f"max_inference_steps: {max_inference_steps}")
        return

    for trial_key in tqdm(trial_keys, desc=json_stem, unit="trial", file=sys.stderr):
        try:
            stimulus = load_left_right_stimulus(
                str(stimulus_path),
                pixel_density=pixel_density,
                skip_t=skip_t,
                trial_number=trial_key,
                inject_occlusion=occlusion_regions,
            )
            # FPS/duration sanity: time axis in plot is frame_index / (fps/skip_t) -> seconds
            duration_sec = stimulus.num_frames * stimulus.skip_t / stimulus.fps
            if verbose:
                print(
                    f"[{trial_key}] num_frames={stimulus.num_frames} fps={stimulus.fps} skip_t={stimulus.skip_t} -> duration={duration_sec:.3f}s",
                    file=sys.stderr,
                )
            assert stimulus.num_frames <= max_inference_steps, (
                f"trial {trial_key}: stimulus.num_frames {stimulus.num_frames} > max_inference_steps {max_inference_steps}"
            )
            model_input.prepare_scene_geometry(stimulus)
            jtap_data, _ = run_parallel_jtap(
                num_jtap_runs,
                key_seed,
                model_input,
                ESS_proportion,
                stimulus,
                num_particles,
                max_inference_steps=max_inference_steps,
            )
            beliefs = jtap_compute_beliefs(jtap_data)
            # Ensure beliefs length matches stimulus so time axis is correct
            arr = beliefs.model_beliefs
            n_belief = arr.shape[1] if arr.ndim == 3 else arr.shape[0]
            assert n_belief == stimulus.num_frames, (
                f"trial {trial_key}: beliefs length {n_belief} != stimulus.num_frames {stimulus.num_frames}"
            )
            fig = jtap_plot_lr_lines(
                lr_beliefs=beliefs,
                stimulus=stimulus,
                show="model",
                include_baselines=False,
                include_start_frame=True,
                return_fig=True,
                include_stimulus=True,
            )
            png_path = output_dir / f"{json_stem}_{trial_key}.png"
            fig.savefig(png_path, dpi=save_dpi)
            plt.close(fig)
        except Exception as e:
            print(f"Failed trial {trial_key}: {e}", file=sys.stderr)


def _cli():
    parser = argparse.ArgumentParser(description="Run JTAP on a single stimulus JSON and save LR plot PNGs per trial.")
    parser.add_argument("stimulus_path", nargs="?", default=None, help="Path to stimulus JSON (or set in config)")
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory for PNGs")
    parser.add_argument("--pixel-density", type=int, default=None)
    parser.add_argument("--skip-t", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--ESS-proportion", type=float, default=None)
    parser.add_argument("--num-jtap-runs", type=int, default=None)
    parser.add_argument("--key-seed", type=int, default=None)
    parser.add_argument(
        "--occlusion-regions",
        type=str,
        default=None,
        help="Occlusion as 'x_left,length' or space-separated pairs, e.g. '4.1,3.2' or '4.1,3.2 5.7,3.2'",
    )
    parser.add_argument("--save-dpi", type=int, default=None, help="PNG DPI for fast save (default from config)")
    parser.add_argument("--dry-run", action="store_true", help="Print trial keys and max_frames only")
    parser.add_argument("--limit-trials", type=int, default=None, help="Process only first N trials")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log FPS, num_frames, duration per trial")
    parser.add_argument(
        "--trial-list",
        type=Path,
        default=None,
        help="Path to file listing trial keys to run (one per line; # comment, blank lines skipped). Only these trials are run; keys not in the JSON are skipped.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.stimulus_path is None:
        args.stimulus_path = config.get("stimulus_path")
    if args.stimulus_path is None:
        parser.error("stimulus_path is required (positional or in config)")
    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "jtap_plots")

    run(
        stimulus_path=args.stimulus_path,
        output_dir=args.output_dir,
        config_path=args.config,
        pixel_density=args.pixel_density,
        skip_t=args.skip_t,
        num_particles=args.num_particles,
        ESS_proportion=args.ESS_proportion,
        num_jtap_runs=args.num_jtap_runs,
        key_seed=args.key_seed,
        occlusion_regions=args.occlusion_regions,
        save_dpi=args.save_dpi,
        dry_run=args.dry_run,
        limit_trials=args.limit_trials,
        verbose=args.verbose,
        trial_list=args.trial_list,
    )


if __name__ == "__main__":
    _cli()
