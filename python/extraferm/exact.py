from typing import Sequence, Union

import numpy as np
from qiskit.circuit import QuantumCircuit

from . import _lib as _rust
from .utils import extract_circuit_data


def exact_calculation(
    *,
    circuit: QuantumCircuit,
    outcome_states: Union[int, Sequence[int]],
) -> Union[float, np.ndarray]:
    """
    Exact calculation of circuit outcome probabilities by summing over all
    controlled-phase masks.

    Args:
        circuit: QuantumCircuit object
        outcome_states: Single state (int) or sequence of states

    Returns:
        - float if a single int is passed,
        - numpy.ndarray of floats if a sequence is passed.
    """
    circuit_data = extract_circuit_data(circuit)

    num_qubits = circuit_data.num_qubits
    normalized_angles = circuit_data.normalized_angles
    initial_state = circuit_data.initial_state
    gate_types = circuit_data.gate_types
    params = circuit_data.params
    qubits = circuit_data.qubits
    orb_indices = circuit_data.orb_indices
    orb_mats = circuit_data.orb_mats

    if isinstance(outcome_states, int):
        out_list = [outcome_states]
        single = True
    else:
        out_list = list(outcome_states)
        single = False

    result_array = _rust.exact_calculation(
        num_qubits,
        normalized_angles,
        initial_state,
        out_list,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    )

    result = np.asarray(result_array)
    if single:
        return float(result[0])
    return result
