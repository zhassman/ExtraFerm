import math
import numpy as np
from numpy.testing import assert_allclose
from passive_extended_matchgate_simulator.raw_estimation import raw_estimate
from passive_extended_matchgate_simulator.utils import (ucj_to_compatible, 
                                                        ucj_to_compatible_fully_reduced, 
                                                        get_bitstrings_and_probs, 
                                                        make_parameterized_controlled_phase_circuit)


def test_mean_0_six_qubits():
    mean, var = 0, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)
    

def test_mean_pi_over_4_six_qubits():
    mean, var = math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_pi_over_2_six_qubits():
    mean, var = math.pi/2, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_3pi_over_4_six_qubits():
    mean, var = 3*math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_pi_six_qubits():
    mean, var = math.pi, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_six_qubits():
    mean, var = -math.pi, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_3pi_over_4_six_qubits():
    mean, var = -3*math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_over_2_six_qubits():
    mean, var = -math.pi/2, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_over_4_six_qubits():
    mean, var = -math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_0_twelve_qubits():
    mean, var = 0, .1
    epsilon, delta, p = .1, .01, 1
    norb, nelec = 6, (3,3)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    first_10_bitstrings = bitstrings[:10]
    first_10_exact_probabilities = exact_probs[:10]
    

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                     outcome_state=first_10_bitstrings, 
                                     epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, first_10_exact_probabilities, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=first_10_bitstrings, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p)
    assert_allclose(probs_compatible_fully_reduced, first_10_exact_probabilities, rtol=0, atol=0.05)


def test_126_qubit_circuit():
    mean, var = 0, .00001
    epsilon, delta, p = .1, .01, 1
    norb, nelec = 63, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var, reduced_interaction=True)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    first_5_bitstrings = bitstrings[:5]
    first_5_exact_probabilities = exact_probs[:5]

    compatible = ucj_to_compatible(circuit)
    probs_compatible = raw_estimate(circuit=compatible,
                                    outcome_state=first_5_bitstrings, 
                                    epsilon=epsilon, delta=delta, p=p)
    assert_allclose(probs_compatible, first_5_exact_probabilities, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = raw_estimate(circuit=compatible_fully_reduced, 
                                                outcome_state=first_5_bitstrings, 
                                                epsilon=epsilon, 
                                                delta=delta, 
                                                p=p)
    assert_allclose(probs_compatible_fully_reduced, first_5_exact_probabilities, rtol=0, atol=0.05)