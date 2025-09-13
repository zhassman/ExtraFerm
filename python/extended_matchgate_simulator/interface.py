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
    epsilon: Optional[float] = None,
    delta:   Optional[float] = None,
    p:       Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Unified interface for quantum circuit probability calculation using matchgate simulation.
    
    This function automatically selects the appropriate simulation algorithm based on the
    provided parameters, offering three different approaches for calculating outcome probabilities:
    
    **Algorithm Selection:**
    
    1. **Raw Estimate**: Fixed Monte Carlo sampling with optional trajectory count.
       - Use when: You want to calculate probabilities for a large number of bitstrings quickly
       - Parameters: Provide either trajectory_count OR (epsilon, delta, p)
       - Performance: Fastest for multiple bitstrings - O(trajectory_count)
    
    2. **Adaptive Estimate**: Uses adaptive Monte Carlo with theoretical error bounds.
       - Use when: You want very high accuracy for fewer bitstrings
       - Parameters: Provide epsilon and delta
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
        epsilon: Target additive error bound (for estimate/raw estimate)
        delta: Failure probability (for estimate/raw estimate) 
        p: Upper bound on outcome probability (for raw estimate only)
    
    Returns:
        float: Probability for single outcome_states
        np.ndarray: Array of probabilities for multiple outcome_states
    
    Examples:
        # Exact calculation
        prob = calculate_probability(circuit=qc, outcome_states=0b1010)
        
        # Adaptive estimate with error bounds
        prob = calculate_probability(circuit=qc, outcome_states=[0b1010, 0b0101], epsilon=0.01, delta=0.05)
        
        # Raw estimate with fixed trajectory count
        prob = calculate_probability(circuit=qc, outcome_states=0b1010, trajectory_count=10000)
        
        # Raw estimate with accuracy bounds
        prob = calculate_probability(circuit=qc, outcome_states=[0b1010, 0b0101], epsilon=0.01, delta=0.05, p=0.1)
    """
    
    # Parameter validation
    if epsilon is not None and delta is None:
        raise ValueError("delta is required when epsilon is provided")
    if delta is not None and epsilon is None:
        raise ValueError("epsilon is required when delta is provided")
    if p is not None and (epsilon is None or delta is None):
        raise ValueError("p can only be used with epsilon and delta (for raw estimate)")
    
    # Check for conflicting parameter combinations
    has_trajectory_count = trajectory_count is not None
    has_accuracy_params = epsilon is not None and delta is not None and p is not None
    has_adaptive_params = epsilon is not None and delta is not None and p is None
    
    if has_trajectory_count and has_accuracy_params:
        raise ValueError("Cannot provide both trajectory_count and (epsilon, delta, p) - choose one approach")
    if has_trajectory_count and has_adaptive_params:
        raise ValueError("Cannot provide both trajectory_count and (epsilon, delta) - choose one approach")
    if has_accuracy_params and has_adaptive_params:
        raise ValueError("Cannot provide both (epsilon, delta, p) and (epsilon, delta) - choose one approach")
    
    if trajectory_count is not None and trajectory_count <= 0:
        raise ValueError("trajectory_count must be positive")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta is not None and (delta <= 0 or delta >= 1):
        raise ValueError("delta must be between 0 and 1")
    if p is not None and (p <= 0 or p > 1):
        raise ValueError("p must be between 0 and 1")
    
    if trajectory_count is not None or (epsilon is not None and delta is not None and p is not None):
        if is_lucj(circuit):
            return raw_estimate_lucj(
                circuit=circuit,
                outcome_states=outcome_states,
                trajectory_count=trajectory_count,
                epsilon=epsilon,
                delta=delta,
                p=p
            )
        else:
            return raw_estimate(
                circuit=circuit,
                outcome_states=outcome_states,
                trajectory_count=trajectory_count,
                epsilon=epsilon,
                delta=delta,
                p=p
            )
    elif epsilon is not None and delta is not None:
        return estimate(
            circuit=circuit,
            outcome_states=outcome_states,
            epsilon=epsilon,
            delta=delta
        )
    else:
        return exact_calculation(
            circuit=circuit,
            outcome_states=outcome_states
        )
