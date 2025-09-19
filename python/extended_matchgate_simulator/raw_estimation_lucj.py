from typing import Optional, Union, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit
from .utils import CircuitData, extract_circuit_data, calculate_trajectory_count, is_lucj
from . import _lib as _rust


def raw_estimate_lucj(
    *,
    circuit: Optional[QuantumCircuit]   = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_states: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta:   Optional[float] = None,
    p:       Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    A version of raw_estimate that is optimized to work for 'lucj'
    circuits. That is, any circuit of the form

    X* , orb_rot_jw , CP* , orb_rot_jw.

    This function returns a Monte-Carlo estimate for one or more 
    outcome_states(s). Pass:
      - exactly one of circuit or circuit_data, and
      - exactly one of (trajectory_count) or (epsilon,delta,p).

    If 'trajectory_count' is provided, it is used directly.
    Otherwise it's computed from (epsilon,delta,p,extent).

    Returns a float for a single state, or an ndarray for multiple.
    """
    assert is_lucj(circuit), "Circuit does not have the correct form."

    if (circuit is None) == (circuit_data is None):
        raise ValueError("Must pass exactly one of 'circuit' or 'circuit_data'")

    if circuit_data is None:
        circuit_data = extract_circuit_data(circuit)

    num_qubits = circuit_data.num_qubits
    extent = circuit_data.extent
    negative_mask = circuit_data.negative_mask
    normalized_angles = circuit_data.normalized_angles
    initial_state = circuit_data.initial_state
    gate_types = circuit_data.gate_types
    params = circuit_data.params
    qubits = circuit_data.qubits
    orb_indices = circuit_data.orb_indices
    orb_mats = circuit_data.orb_mats

    accuracy_args = (epsilon is not None) and (delta is not None) and (p is not None)
    if (trajectory_count is None) == (not accuracy_args):
        raise ValueError(
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' but not both."
        )

    t = trajectory_count if (trajectory_count is not None) else calculate_trajectory_count(epsilon, delta, p, extent)

    if isinstance(outcome_states, int):
        return _rust.raw_estimate_lucj_single(
                num_qubits,
                normalized_angles,
                negative_mask,
                extent,
                initial_state,
                outcome_states,
                t,
                gate_types,
                params,
                qubits,
                orb_indices,
                orb_mats,
            )
    else:
        return _rust.raw_estimate_lucj_batch(
                num_qubits,
                normalized_angles,
                negative_mask,
                extent,
                initial_state,
                outcome_states,
                t,
                gate_types,
                params,
                qubits,
                orb_indices,
                orb_mats,
            )