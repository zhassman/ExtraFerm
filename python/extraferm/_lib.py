try:
    from ._lib import (
        estimate_batch,
        estimate_single,
        exact_calculation,
        raw_estimate_batch,
        raw_estimate_lucj_batch,
        raw_estimate_lucj_single,
        raw_estimate_reuse,
        raw_estimate_single,
    )
except ImportError as e:
    raise ImportError(
        "Rust extension module not found. "
        "Please ensure the backend has been built successfully."
    ) from e

__all__ = [
    "estimate_batch",
    "estimate_single",
    "exact_calculation",
    "raw_estimate_batch",
    "raw_estimate_lucj_batch",
    "raw_estimate_lucj_single",
    "raw_estimate_reuse",
    "raw_estimate_single",
]
