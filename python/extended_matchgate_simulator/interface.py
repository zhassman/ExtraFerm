from .exact import exact_calculation
from .raw_estimation import raw_estimate
from .raw_estimation_lucj import raw_estimate_lucj
from .utils import is_lucj
from .estimation import estimate


def calculate_probabilities():
    # no trajectory count / no epsilon delta --> estimate
    # everything provided --> raw_estimate
    # everything provided and LUCJ --> raw_estimate_lucj
    # either delta or epsilon = 0 --> exact