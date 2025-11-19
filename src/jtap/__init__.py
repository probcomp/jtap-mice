__all__ = ["model", "stimuli", "viz", "utils", "distributions", "core", "evaluation"]

# Import all submodules into the package namespace
from . import model, viz, utils, inference, distributions, core, evaluation

import os
import treescope
import warnings
from matplotlib import font_manager, rcParams
from .utils import get_fonts_dir
from scipy.stats import (
    NearConstantInputWarning, ConstantInputWarning
)

# Warning filters
warnings.filterwarnings("ignore", category=NearConstantInputWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Font setup
font_path = os.path.join(get_fonts_dir(), "DMSans-Regular.ttf")
dm_sans_prop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
font_manager.fontManager.addfont(os.path.join(get_fonts_dir(), "DMSans-Bold.ttf"))
rcParams['font.family'] = dm_sans_prop.get_name()


JTAP_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def pretty():
    treescope.register_as_default()
    treescope.register_autovisualize_magic()
    treescope.active_autovisualizer.set_interactive(treescope.ArrayAutovisualizer())


# Automatically call pretty() if running in an IPython notebook
try:
    from IPython import get_ipython

    shell = get_ipython()
    if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
        pretty()
except Exception:
    pass


def set_jaxcache():
    import jax

    cache_dir = os.path.join(JTAP_ROOT, ".jax_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.experimental.compilation_cache.compilation_cache.set_cache_dir(cache_dir)