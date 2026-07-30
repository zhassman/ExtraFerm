from extraferm import outcome_probabilities
import ffsim
import qiskit
from unittest import TestCase
import pytest


@pytest.fixture
def circ():
    """
    Creates 2-orbital fermionic circuit in the HF state with one orbital rotation
    """
    norb = 2
    qubits = qiskit.QuantumRegister(norb * 2)  # (2e,2o)
    circ = qiskit.QuantumCircuit(qubits)

    circ.append(ffsim.qiskit.PrepareHartreeFockJW(norb, (1, 1)), qubits)
    circ.append(
        ffsim.qiskit.OrbitalRotationJW(
            norb, ffsim.random.random_unitary(norb, seed=314)
        ),
        qubits,
    )
    circ.measure_all()

    return circ


def test_hartree_fock_jw(circ):
    """
    Tests whether outcome_probabilities can handle global hartree_fock_jw gates
    """
    # print(circ.count_ops())
    probs = outcome_probabilities(circuit=circ, outcome_states=0b1010)


def test_slater_jw(circ):
    circ = circ.decompose(["hartree_fock_jw"])
    # print(circ.count_ops())
    probs = outcome_probabilities(circuit=circ, outcome_states=0b1010)


def test_global_phase(circ):
    circ = circ.decompose(["hartree_fock_jw"])
    circ = circ.decompose(["slater_jw"])
    # print(circ.count_ops())
    probs = outcome_probabilities(circuit=circ, outcome_states=0b1010)


def test_barrier(circ):
    circ = circ.decompose(["hartree_fock_jw"])
    circ = circ.decompose(["slater_jw"])
    # Strip the global phase gate
    circ = qiskit.QuantumCircuit.from_instructions([inst for inst in circ.data if inst.operation.name != "global_phase"])
    # print(circ.count_ops())
    probs = outcome_probabilities(circuit=circ, outcome_states=0b1010)


def test_x_gates(circ):
    circ = circ.decompose(["hartree_fock_jw"])
    circ = circ.decompose(["slater_jw"])
    # print(circ.count_ops())
    probs = outcome_probabilities(circuit=circ, outcome_states=0b1010)

# TODO: X gates
