from typing import Optional, Union, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit
from . import _lib as _rust
from .utils import extract_circuit_data, CircuitData

def estimate(
    *,
    circuit: Optional[QuantumCircuit] = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_states: Union[int, Sequence[int]],
    epsilon: float,
    delta: float,
) -> Union[float, np.ndarray]:
    """
    Estimate probability for one or more outcome states using adaptive Monte Carlo.
    Pass exactly one of circuit or circuit_data.
    
    Args:
        circuit: QuantumCircuit object (optional)
        circuit_data: Pre-extracted circuit data (optional)
        outcome_states: Single state (int) or sequence of states
        epsilon: Target accuracy
        delta: Confidence parameter
        
    Returns:
        Float for single state, ndarray for multiple states.
        Multiple states are computed in parallel for efficiency.
    """
    if (circuit is None) == (circuit_data is None):
        raise ValueError("Provide exactly one of circuit or circuit_data")
    if circuit_data is None:
        circuit_data = extract_circuit_data(circuit)

    (
        num_qubits,
        extent,
        negative_mask,
        angles,
        initial_state,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    ) = circuit_data

    if isinstance(outcome_states, int):
        return _rust.estimate_single(
            num_qubits,
            angles,
            negative_mask,
            extent,
            initial_state,
            outcome_states,
            epsilon,
            delta,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
        )
    else:
        return _rust.estimate_batch(
            num_qubits,
            angles,
            negative_mask,
            extent,
            initial_state,
            outcome_states,
            epsilon,
            delta,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
        )
