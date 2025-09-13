from .exact import exact_calculation
from .raw_estimation import raw_estimate
from .raw_estimation_lucj import raw_estimate_lucj
from .utils import is_lucj
from .estimation import estimate
from typing import Union, Optional, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit


def calculate_probability(
    *,
    circuit: QuantumCircuit,
    outcome_states: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    additive_error: Optional[float] = None,
    failure_probability: Optional[float] = None,
    probability_upper_bound: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Unified interface for quantum circuit probability calculation using extended matchgate simulation.
    
    This function automatically selects the appropriate simulation algorithm based on the
    provided parameters, offering three different approaches for calculating outcome probabilities:
    
    **Algorithm Selection:**
    
    1. **Raw Estimate**: Fixed Monte Carlo sampling with optional trajectory count.
       - Use when: You want to calculate probabilities for a large number of bitstrings quickly
       - Parameters: Provide either trajectory_count OR (additive_error, failure_probability, probability_upper_bound)
       - Performance: Fastest for multiple bitstrings - O(trajectory_count)
    
    2. **Adaptive Estimate**: Uses adaptive Monte Carlo with theoretical error bounds.
       - Use when: You want very high accuracy for fewer bitstrings
       - Parameters: Provide additive_error and failure_probability
       - Performance: Adaptive - automatically determines sample size for guaranteed accuracy
    
    3. **Exact Calculation**: Returns exact probabilities by summing over all controlled-phase masks.
       - Use when: You need exact results and the circuit is small enough
       - Parameters: Only circuit and outcome_states required
       - Performance: O(2^k) where k is the number of controlled-phase gates
    
    **Automatic Optimizations:**
    - LUCJ circuits (X*, orb_rot_jw, CP*, orb_rot_jw pattern) automatically use optimized routines
    
    Args:
        circuit: QuantumCircuit compatible with matchgate simulation
        outcome_states: Single outcome (int) or sequence of outcomes to calculate
        trajectory_count: Fixed number of Monte Carlo trajectories (for raw estimate)
        additive_error: Target additive error bound (for estimate/raw estimate)
        failure_probability: Failure probability (for estimate/raw estimate) 
        probability_upper_bound: Upper bound on outcome probability (for raw estimate only)
    
    Returns:
        float: Probability for single outcome_states
        np.ndarray: Array of probabilities for multiple outcome_states
    
    Examples:
        # Exact calculation
        prob = calculate_probability(circuit=qc, outcome_states=0b1010)
        
        # Adaptive estimate with error bounds
        prob = calculate_probability(circuit=qc, outcome_states=[0b1010, 0b0101], additive_error=0.01, failure_probability=0.05)
        
        # Raw estimate with fixed trajectory count
        prob = calculate_probability(circuit=qc, outcome_states=0b1010, trajectory_count=10000)
        
        # Raw estimate with accuracy bounds
        prob = calculate_probability(circuit=qc, outcome_states=[0b1010, 0b0101], additive_error=0.01, failure_probability=0.05, probability_upper_bound=0.1)
    """
    
    # Parameter validation
    if additive_error is not None and failure_probability is None:
        raise ValueError("failure_probability is required when additive_error is provided")
    if failure_probability is not None and additive_error is None:
        raise ValueError("additive_error is required when failure_probability is provided")
    if probability_upper_bound is not None and (additive_error is None or failure_probability is None):
        raise ValueError("probability_upper_bound can only be used with additive_error and failure_probability (for raw estimate)")
    
    # Check for conflicting parameter combinations
    has_trajectory_count = trajectory_count is not None
    has_accuracy_params = additive_error is not None and failure_probability is not None and probability_upper_bound is not None
    has_adaptive_params = additive_error is not None and failure_probability is not None and probability_upper_bound is None
    
    if has_trajectory_count and has_accuracy_params:
        raise ValueError("Cannot provide both trajectory_count and (additive_error, failure_probability, probability_upper_bound) - choose one approach")
    if has_trajectory_count and has_adaptive_params:
        raise ValueError("Cannot provide both trajectory_count and (additive_error, failure_probability) - choose one approach")
    if has_accuracy_params and has_adaptive_params:
        raise ValueError("Cannot provide both (additive_error, failure_probability, probability_upper_bound) and (additive_error, failure_probability) - choose one approach")
    
    if trajectory_count is not None and trajectory_count <= 0:
        raise ValueError("trajectory_count must be positive")
    if additive_error is not None and additive_error <= 0:
        raise ValueError("additive_error must be positive")
    if failure_probability is not None and (failure_probability <= 0 or failure_probability >= 1):
        raise ValueError("failure_probability must be between 0 and 1")
    if probability_upper_bound is not None and (probability_upper_bound <= 0 or probability_upper_bound > 1):
        raise ValueError("probability_upper_bound must be between 0 and 1")
    
    if trajectory_count is not None or (additive_error is not None and failure_probability is not None and probability_upper_bound is not None):
        if is_lucj(circuit):
            return raw_estimate_lucj(
                circuit=circuit,
                outcome_states=outcome_states,
                trajectory_count=trajectory_count,
                epsilon=additive_error,
                delta=failure_probability,
                p=probability_upper_bound
            )
        else:
            return raw_estimate(
                circuit=circuit,
                outcome_states=outcome_states,
                trajectory_count=trajectory_count,
                epsilon=additive_error,
                delta=failure_probability,
                p=probability_upper_bound
            )
    elif additive_error is not None and failure_probability is not None:
        return estimate(
            circuit=circuit,
            outcome_states=outcome_states,
            epsilon=additive_error,
            delta=failure_probability
        )
    else:
        return exact_calculation(
            circuit=circuit,
            outcome_states=outcome_states
        )
