from typing import Optional, Sequence, Union

import numpy as np
from qiskit.circuit import QuantumCircuit

from . import _lib as _rust
from .utils import calculate_trajectory_count, extract_circuit_data, is_lucj


def raw_estimate(
    *,
    circuit: QuantumCircuit,
    outcome_states: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    p: Optional[float] = None,
    reuse_trajectories: Optional[bool] = False,
    seed: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Monte-Carlo estimate for one or more outcome_states(s). Pass exactly one of:
      - trajectory_count, or
      - (epsilon, delta, p).

    If 'trajectory_count' is provided, it is used directly.
    Otherwise it's computed from (epsilon, delta, p, extent).

    If reuse_trajectories is set to True, then the Rust backend will
    use the same pool of trajectories to calculate probabilities for
    all bitstrings.

    If 'seed' is provided, it will be used to seed the random number generator
    for reproducible results. If None, a random seed will be used.

    Returns a float for a single state, or an ndarray for multiple.
    """
    if reuse_trajectories and isinstance(outcome_states, int):
        raise ValueError(
            "'reuse_trajectories=True' only makes sense when 'outcome_states' is a "
            "sequence"
        )

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
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' but not "
            "both."
        )

    t = (
        trajectory_count
        if (trajectory_count is not None)
        else calculate_trajectory_count(epsilon, delta, p, extent)
    )

    if seed is None:
        import random

        seed = random.getrandbits(64)

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
            seed,
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
                seed,
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
                seed,
            )


def raw_estimate_lucj(
    *,
    circuit: Optional[QuantumCircuit] = None,
    outcome_states: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    p: Optional[float] = None,
    seed: Optional[int] = None,
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

    If 'seed' is provided, it will be used to seed the random number generator
    for reproducible results. If None, a random seed will be used.

    Returns a float for a single state, or an ndarray for multiple.
    """
    assert is_lucj(circuit), "Circuit does not have the correct form."

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
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' but not "
            "both."
        )

    t = (
        trajectory_count
        if (trajectory_count is not None)
        else calculate_trajectory_count(epsilon, delta, p, extent)
    )

    if seed is None:
        import random

        seed = random.getrandbits(64)

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
            seed,
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
            seed,
        )
