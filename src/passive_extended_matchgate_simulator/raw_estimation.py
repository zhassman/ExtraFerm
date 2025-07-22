from typing import Optional, Union, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit
from . import utils
from .utils import CircuitData, extract_circuit_data, calculate_trajectory_count
import emsim as _rust


def raw_estimate(
    *,
    circuit: Optional[QuantumCircuit]   = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_state: Union[int, Sequence[int]],
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta:   Optional[float] = None,
    p:       Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Monte-Carlo estimate for one or more outcome_state(s). Pass:
      - exactly one of circuit or circuit_data, and
      - exactly one of (trajectory_count) or (epsilon,delta,p).

    If 'trajectory_count' is provided, it is used directly.
    Otherwise it's computed from (epsilon,delta,p,extent).

    Returns a float for a single state, or an ndarray for multiple.
    """

    # 1) load or extract your circuit_data
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

    # 2) trajectory count vs (epsilon,delta,p)
    accuracy_args = (epsilon is not None) and (delta is not None) and (p is not None)
    if (trajectory_count is None) == (not accuracy_args):
        raise ValueError(
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' but not both."
        )
    t = trajectory_count if (trajectory_count is not None) else calculate_trajectory_count(epsilon, delta, extent, p)

    # 3) dispatch to Rust
    if isinstance(outcome_state, int):
        # single-state path
        return _rust.raw_estimate_single(
            num_qubits,
            normalized_angles,
            negative_mask,
            extent,
            initial_state,
            outcome_state,   # a single Python int → Rust u128
            t,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
        )
    else:
        # batch path: any sequence of Python ints (up to 128 bits)
        # we pass the Python iterable directly; Rust will extract u128
        return _rust.raw_estimate_batch(
            num_qubits,
            normalized_angles,
            negative_mask,
            extent,
            initial_state,
            outcome_state,   # sequence of Python ints → Rust Vec<u128>
            t,
            gate_types,
            params,
            qubits,
            orb_indices,
            orb_mats,
        )
