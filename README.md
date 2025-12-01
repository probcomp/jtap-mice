### Cog-Neuro Modeling of Joint Tracking and Prediction in Mice

`jtap-mice` is a Python library for probabilistic cog-neuro modeling and inference of the left-right task designed for Mice.

## Installation

#### Requirements

- Python 3.11 or higher
- For GPU acceleration: NVIDIA GPU with CUDA 12 (recommended) or CUDA 11
- For CPU-only: Any system (including macOS, Linux, Windows)

#### Setup

Follow the steps below to get `jtap` running on your machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/probcomp/jtap.git
   cd jtap
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

4. **Verify your installation**:
   ```bash
   uv run python -c "import jax; print('JAX devices:', jax.devices())"
   ```

#### Platform Compatibility

- **NVIDIA GPU (Linux/Windows)**: Use `uv sync --extra cuda` for best performance
- **Apple Silicon (M1/M2/M3 Macs)**: Use `uv sync --extra cpu` - JAX will automatically use Metal acceleration
- **Intel/AMD CPU (any OS)**: Use `uv sync --extra cpu` - works on all systems

**Note**: The CPU version will work on any system, including those with GPUs, but will be slower than GPU-accelerated versions.
