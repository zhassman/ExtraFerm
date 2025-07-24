from typing import Optional, Sequence, Union
import numpy as np
from qiskit.circuit import QuantumCircuit
from .utils import CircuitData, extract_circuit_data
import emsim as _rust


def exact_calculation(
    *,
    circuit: Optional[QuantumCircuit] = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_states: Union[int, Sequence[int]],
) -> Union[float, np.ndarray]:
    """
    Exact calculation of circuit outcome probabilities by summing over all controlled-phase masks.

    Pass exactly one of 'circuit' or 'circuit_data', and one 'outcome_states' (int or sequence of ints).

    Returns:
        - float if a single int is passed,
        - numpy.ndarray of floats if a sequence is passed.
    """
    if (circuit is None) == (circuit_data is None):
        raise ValueError("Must pass exactly one of 'circuit' or 'circuit_data'")

    if circuit_data is None:
        circuit_data = extract_circuit_data(circuit)

    (
        num_qubits,
        _extent,
        _negative_mask,
        normalized_angles,
        initial_state,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    ) = circuit_data

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
