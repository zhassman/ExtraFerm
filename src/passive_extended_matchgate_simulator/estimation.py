from typing import Optional
from qiskit.circuit import QuantumCircuit
import emsim as _rust
from .utils import extract_circuit_data, CircuitData

def estimate(
    *,
    circuit: Optional[QuantumCircuit] = None,
    circuit_data: Optional[CircuitData] = None,
    outcome_state: int,
    epsilon: float,
    delta: float,
) -> float:
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

    return _rust.estimate(
        num_qubits,
        angles,
        negative_mask,
        extent,
        initial_state,
        outcome_state,
        epsilon,
        delta,
        gate_types,
        params,
        qubits,
        orb_indices,
        orb_mats,
    )
