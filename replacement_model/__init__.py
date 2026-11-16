"""SSC pipeline V3 (scale-specific contributors): Bands -> S_orig -> per-gene removal test on S_orig -> substitution table."""
from .pipeline import run_v3, write_outputs, select_dictionary, SIGNATURE
from .v3lib import minimal_panel
__all__ = ["run_v3", "write_outputs", "select_dictionary", "minimal_panel", "SIGNATURE"]
