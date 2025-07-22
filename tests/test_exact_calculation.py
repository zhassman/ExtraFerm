import math
from numpy.testing import assert_allclose
from passive_extended_matchgate_simulator.exact import exact_calculation
from passive_extended_matchgate_simulator.utils import (ucj_to_compatible, 
                                                        ucj_to_compatible_fully_reduced, 
                                                        get_bitstrings_and_probs, 
                                                        make_parameterized_controlled_phase_circuit)


def test_mean_0_six_qubits():
    mean, var = 0, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_pi_over_4_six_qubits():
    mean, var = math.pi/4, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_pi_over_2_six_qubits():
    mean, var = math.pi/2, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_3pi_over_4_six_qubits():
    mean, var = 3*math.pi/4, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_pi_six_qubits():
    mean, var = math.pi, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)



def test_mean_negative_pi_six_qubits():
    mean, var = -math.pi, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_negative_3pi_over_4_six_qubits():
    mean, var = -3*math.pi/4, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_negative_pi_over_2_six_qubits():
    mean, var = -math.pi/2, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)


def test_mean_negative_pi_over_4_six_qubits():
    mean, var = -math.pi/4, .1
    norb, nelec = 3, (1,1)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)



def test_mean_pi_twelve_qubits():
    mean, var = math.pi, .1
    norb, nelec = 6, (3,3)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)



def test_mean_0_twelve_qubits():
    mean, var = 0, .1
    norb, nelec = 6, (3,3)

    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=False)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)

    compatible = ucj_to_compatible(circuit)
    probs_compatible = exact_calculation(circuit=compatible, outcome_states=bitstrings)
    assert_allclose(probs_compatible, exact_probs)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = exact_calculation(circuit=compatible_fully_reduced, outcome_states=bitstrings)
    assert_allclose(probs_compatible_fully_reduced, exact_probs)
