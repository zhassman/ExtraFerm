from typing import Optional, Sequence, Union

import numpy as np
from qiskit.circuit import QuantumCircuit

from . import _lib as _rust
from .utils import extract_circuit_data, is_lucj


def estimate(
    *,
    circuit: QuantumCircuit,
    outcome_states: Union[int, Sequence[int]],
    epsilon: float,
    delta: float,
    seed: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Estimate probability for one or more outcome states using adaptive Monte Carlo.

    Args:
        circuit: QuantumCircuit object
        outcome_states: Single state (int) or sequence of states
        epsilon: Target accuracy
        delta: Confidence parameter
        seed: Optional seed for reproducible results. If None, a random seed will be
        used.

    Returns:
        Float for single state, ndarray for multiple states.
        Multiple states are computed in parallel for efficiency.
    """
    circuit_data = extract_circuit_data(circuit)
    use_lucj = is_lucj(circuit)

    num_qubits = circuit_data.num_qubits
    extent = circuit_data.extent
    negative_mask = circuit_data.negative_mask
    angles = circuit_data.normalized_angles
    initial_state = circuit_data.initial_state
    gate_types = circuit_data.gate_types
    params = circuit_data.params
    qubits = circuit_data.qubits
    orb_indices = circuit_data.orb_indices
    orb_mats = circuit_data.orb_mats

    if seed is None:
        import random

        seed = random.getrandbits(64)

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
            use_lucj,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
            seed,
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
            use_lucj,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
            seed,
        )
