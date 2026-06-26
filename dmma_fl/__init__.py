"""DMMA-FL reproduction package.

The torch-heavy trainer is intentionally not imported here so environment
utilities can run on bare servers without PyTorch installed.
"""

from .config import DMMAConfig
from .decomposition import simplex_weights

__all__ = ["DMMAConfig", "simplex_weights"]
