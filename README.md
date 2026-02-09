### Cog-Neuro Modeling of Joint Tracking and Prediction in Mice

`jtap-mice` is a Python library for probabilistic cog-neuro modeling and inference of the left-right task designed for Mice.

## Installation

#### Requirements

- Python 3.11 or higher
- For GPU acceleration: NVIDIA GPU with CUDA 12 (recommended) or CUDA 11
- For CPU-only: Any system (including macOS, Linux, Windows)

#### Setup

**Do not run this inside a conda or a virtual environment (venv, virtualenv or pyenv. Not even pixi) UV is meant to replace all of this**

Follow the steps below to get `jtap` running on your machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/probcomp/jtap-mice.git
   cd jtap-mice
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install JTAP**:

   **For NVIDIA GPU systems (Linux/Windows with CUDA 12):**
   ```bash
   uv sync --extra cuda
   ```

   **For CPU-only systems (macOS, systems without NVIDIA GPU):**
   ```bash
   uv sync --extra cpu
   ```
4. **Install Rerun Locally (If not on a VM and you have a Display)**:

```bash
uv run rerun
```

1. **Verify your installation**:
   ```bash
   uv run python -c "import jax; print('JAX devices:', jax.devices())"
   ```

#### Platform Compatibility

- **NVIDIA GPU (Linux/Windows)**: Use `uv sync --extra cuda` for best performance
- **Apple Silicon (M1/M2/M3 Macs)**: Use `uv sync --extra cpu` - JAX will automatically use Metal acceleration
- **Intel/AMD CPU (any OS)**: Use `uv sync --extra cpu` - works on all systems

**Note**: The CPU version will work on any system, including those with GPUs, but will be slower than GPU-accelerated versions.

---

## Running JTAP on a stimulus JSON

The script `scripts/run_jtap_single_json.py` runs JTAP on **all trials** in a single stimulus JSON and saves one left-right belief plot (PNG) per trial. Run from the repo root.

**Basic usage**

```bash
# One JSON (output goes to jtap_plots/ by default)
uv run python scripts/run_jtap_single_json.py path/to/stimulus.json

# Custom output directory
uv run python scripts/run_jtap_single_json.py path/to/stimulus.json --output-dir my_plots/

# With occlusion: one occluder "x_left,length", e.g. left occluder 4.1–7.3
uv run python scripts/run_jtap_single_json.py path/to/stimulus.json --occlusion-regions "4.1,3.2"

# Run only trials listed in a file (one key per line; good for 100+ trials)
uv run python scripts/run_jtap_single_json.py path/to/stimulus.json --trial-list path/to/trials.txt
```

**Useful flags**

- `--output-dir`, `-o` — Directory for PNGs (default from `scripts/jtap_run_config.yaml`).
- `--config`, `-c` — Path to YAML config (default: `scripts/jtap_run_config.yaml`).
- `--occlusion-regions` — Occluder(s) as `"x_left,length"` or space-separated pairs, e.g. `"4.1,3.2"` or `"4.1,3.2 5.7,3.2"`.
- `--limit-trials N` — Run only the first N trials.
- `--trial-list PATH` — Run only trials whose keys are listed in the file at `PATH`. File format: one trial key per line (e.g. `x1`, `x2`); blank lines and lines starting with `#` are ignored. Keys in the file that are not in the JSON are skipped (a short warning is printed). Use this for large subsets (e.g. 100+ trials) instead of a long CLI list.
- `--dry-run` — Print trial keys and max_frames only; no JTAP or PNGs.
- `--verbose`, `-v` — Log FPS, num_frames, and duration per trial.

**Config and overrides**

All run parameters live in `scripts/jtap_run_config.yaml`. You can:

1. **Edit the YAML** — Change defaults (e.g. `output_dir`, `num_particles`, `save_dpi`, model params).
2. **Override via CLI** — Any of the options above plus e.g. `--num-particles 50`, `--key-seed 42`, `--save-dpi 80`.
3. **Override via kwargs** — When calling the script’s `run()` from Python, pass the same names as keyword arguments; they override the YAML.

**Kwargs / CLI-overridable parameters**

- **Paths / run:** `stimulus_path`, `output_dir`, `config_path`
- **Stimulus:** `pixel_density`, `skip_t`, `occlusion_regions`
- **JTAP run:** `num_particles`, `ESS_proportion`, `num_jtap_runs`, `key_seed`
- **Plot:** `save_dpi`
- **Control:** `dry_run`, `limit_trials`, `verbose`, `trial_list`

Model parameters (in YAML and overridable via kwargs) include: `model_outlier_prob`, `proposal_direction_outlier_tau_deg`, `proposal_direction_outlier_alpha`, `sigma_pos_model`, `sigma_speed_model`, `model_direction_flip_prob`, `pixel_corruption_prob`, `tile_size`, `sigma_pixel_spatial`, `image_power_beta`, `max_speed`, `max_num_occ`, `num_x_grid`, `grid_size_bounds`, `simulate_every`, `sigma_pos_simulation`, `sigma_speed_simulation`, `simulation_direction_flip_prob`, `sigma_pos_initprop`, `proposal_direction_flip_prob`, `sigma_speed_stepprop`, `sigma_pos_stepprop`. See `scripts/jtap_run_config.yaml` for defaults.

---

### Note for Gabe: running all gabe_v1 stimuli

To run JTAP on all six gabe_v1 JSONs with the correct occlusion per file (noOcc / leftOcc / rightOcc), use the shell script from the **repo root**:

```bash
uv run ./scripts/run_all_gabe_v1.sh
```

The script runs the Python script once per JSON, injects occlusion from the filename (left occluder 4.1–7.3 for leftOcc, right occluder 5.7–8.9 for rightOcc, none for noOcc), and writes PNGs into a **timestamped** directory `gabe_v1_plots_YYYYMMDD_HHMM`. Progress is shown per file and per trial (tqdm).
