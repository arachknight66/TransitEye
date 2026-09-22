"""FITS and archive adapters; this package contains no scientific preprocessing."""

from transiteye.io.fits import FitsValidationError, read_tess_light_curve
from transiteye.io.manifest import ProductManifest

__all__ = ["FitsValidationError", "ProductManifest", "read_tess_light_curve"]
