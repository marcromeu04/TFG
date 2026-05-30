"""Synthetic NMR spectrum generation."""
from synth.generator import use_ranges_override, GeneratorConfig, generate_synthetic, build_residual_library
from synth.ranges import get_ranges
from synth.correlations import get_correlation_matrix

__all__ = [
    "GeneratorConfig",
    "generate_synthetic",
    "build_residual_library",
    "get_ranges",
    "get_correlation_matrix",
]
