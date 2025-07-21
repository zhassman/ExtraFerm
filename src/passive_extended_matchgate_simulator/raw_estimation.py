from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from . import utils
from .utils import CircuitData, extract_circuit_data, calculate_trajectory_count
import emsim as _rust


def raw_estimate(
    *,
    circuit: Optional[QuantumCircuit]   = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_state: int,
    trajectory_count: Optional[int] = None,
    epsilon: Optional[float] = None,
    delta:   Optional[float] = None,
    p:       Optional[float] = None,
) -> float:
    """
    Monte-Carlo estimate for outcome_state.   pass:
      - exactly one of circuit or circuit_data, and
      - exactly one of (trajectory_count) or (epsilon,delta,p).

    If 'trajectory_count' is provided, it is used directly.
    Otherwise it's computed from (epsilon,delta,p,extent).
    """

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
            "Must pass either 'trajectory_count' or all of 'epsilon, delta, p' (but not both)."
        )

    if trajectory_count is not None:
        t = trajectory_count
    else:
        t = calculate_trajectory_count(epsilon, delta, extent, p)


    # # in raw_estimation.py, just above the Rust call:
    # print("  negative_mask:", negative_mask, "  bit_length:", negative_mask.bit_length())
    # print("  initial_state:",    initial_state,    "  bit_length:", initial_state.bit_length())
    # print("  outcome_state:",    outcome_state,    "  bit_length:", outcome_state.bit_length())
    # print("  trajectory_count:", t,                "  bit_length:", t.bit_length())


    return _rust.raw_estimate(
        num_qubits,
        normalized_angles,
        negative_mask,
        extent,
        initial_state,
        outcome_state,
        t,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    )
