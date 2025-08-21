# Fixed preamble.py for Introduction to Machine Learning with Python

try:
    from IPython.display import set_matplotlib_formats, display
except ImportError:
    # For newer versions of IPython/Jupyter
    try:
        from IPython.core.display import set_matplotlib_formats, display
    except ImportError:
        # If still not available, define dummy functions
        def set_matplotlib_formats(*args, **kwargs):
            pass
        from IPython.display import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from cycler import cycler

# Set matplotlib formats (will be skipped if not available)
set_matplotlib_formats('pdf', 'png')

# Configure matplotlib settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.numpoints'] = 1

# Set color cycle
plt.rc('axes', prop_cycle=(
    cycler('color', mglearn.plot_helpers.cm_cycle.colors) +
    cycler('linestyle', ['-', '-', "--", (0, (3, 3)), (0, (1.5, 1.5))])))

# Configure numpy and pandas display options
np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.max_columns", 8)
pd.set_option('display.precision', 2)

# Fixed: Changed **all** to __all__ (double underscore, not asterisk)
__all__ = ['np', 'mglearn', 'display', 'plt', 'pd']
