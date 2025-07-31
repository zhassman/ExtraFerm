from typing import Optional, Union, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit
from .utils import CircuitData, extract_circuit_data, calculate_trajectory_count
from . import _lib as _rust


def raw_estimate(
    *,
    circuit: Optional[QuantumCircuit]   = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_states: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta:   Optional[float] = None,
    p:       Optional[float] = None,
    reuse_trajectories: Optional[bool] = False
) -> Union[float, np.ndarray]:
    """
    Monte-Carlo estimate for one or more outcome_states(s). Pass:
      - exactly one of circuit or circuit_data, and
      - exactly one of (trajectory_count) or (epsilon,delta,p).

    If 'trajectory_count' is provided, it is used directly.
    Otherwise it's computed from (epsilon,delta,p,extent).

    If reuse_trajectories is set to True, then the Rust backend will
    use the same pool of trajectories to calculate probabilities for
    all bitstrings.

    Returns a float for a single state, or an ndarray for multiple.
    """
    if reuse_trajectories and isinstance(outcome_states, int):
        raise ValueError("'reuse_trajectories=True' only makes sense when 'outcome_states' is a sequence")

    if (circuit is None) == (circuit_data is None):
        raise ValueError("Must pass exactly one of 'circuit' or 'circuit_data'")

    if circuit_data is None:
        circuit_data = extract_circuit_data(circuit)

    (
        num_qubits,
        extent,
        negative_mask,
        normalized_angles,
        initial_state,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    ) = circuit_data

    accuracy_args = (epsilon is not None) and (delta is not None) and (p is not None)
    if (trajectory_count is None) == (not accuracy_args):
        raise ValueError(
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' but not both."
        )
    t = trajectory_count if (trajectory_count is not None) else calculate_trajectory_count(epsilon, delta, extent, p)

    if isinstance(outcome_states, int):

        return _rust.raw_estimate_single(
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

        if reuse_trajectories:
            return _rust.raw_estimate_reuse(
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

            return _rust.raw_estimate_batch(
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
