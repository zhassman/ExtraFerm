try:
    from ._lib import *
except ImportError:
    raise ImportError(
        "Rust extension module not found. "
        "Please ensure the package is properly installed with 'pip install -e .'"
    ) 