import math
from typing import Optional

import ffsim
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister


def make_parameterized_controlled_phase_circuit(
    norb: int,
    nelec: tuple[int, int],
    mean: float,
    var: float,
    reduced_interaction: bool = False,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Crates a circuit consisting an orbital rotation followed by a number of
    controlled phase gates followed by another orbital rotation.

    norb: The number of spatial orbitals
    nelec: The number of electrons in spatial orbital
    mean: specifies the average angles of the controlled-phase gates
    var: specifies the variance of the angles of the controlled-phase gates
    """

    if reduced_interaction:
        alpha_alpha_indices = [(p, p + 1) for p in range(0, norb - 1, 4)]
        alpha_beta_indices = [(p, p) for p in range(0, norb, 4)]
    else:
        alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
        alpha_beta_indices = [(p, p) for p in range(norb)]

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    scale = math.sqrt(var)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        with_final_orbital_rotation=False,
        diag_coulomb_mean=mean,
        diag_coulomb_scale=scale,
        diag_coulomb_normal=True,
        seed=seed,
    )
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    return circuit


def get_bitstrings_and_probs(
    circuit: QuantumCircuit, norb: int, nelec: tuple[int, int]
) -> tuple[list[int], list[np.float64]]:
    """
    Returns a pair of lists, the former containing all valid bitstrings
    given by the ffsim statevector simulation and latter containing their
    respective probabilities.

    Args:
        circuit: A quantum circuit compatible with ffsim
        norb: The number of spatial orbitals
        nelec: The number of electrons in spatial orbital
    """

    statevec = ffsim.qiskit.final_state_vector(circuit)
    probs = np.abs(statevec.vec) ** 2
    bitstrings = ffsim.addresses_to_strings(range(len(probs)), norb, nelec)
    return bitstrings, probs
