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
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)
    

def test_mean_pi_over_4_six_qubits():
    mean, var = math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_pi_over_2_six_qubits():
    mean, var = math.pi/2, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_3pi_over_4_six_qubits():
    mean, var = 3*math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_pi_six_qubits():
    mean, var = math.pi, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_six_qubits():
    mean, var = -math.pi, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_3pi_over_4_six_qubits():
    mean, var = -3*math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_over_2_six_qubits():
    mean, var = -math.pi/2, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_negative_pi_over_4_six_qubits():
    mean, var = -math.pi/4, .1

    epsilon, delta, p = .1, .01, 1
    norb, nelec = 3, (1,1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec) 

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings]
    assert_allclose(probs_compatible, exact_probs, rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings]
    assert_allclose(probs_compatible_fully_reduced, exact_probs, rtol=0, atol=0.05)


def test_mean_0_twelve_qubits():
    mean, var = 0, .1
    epsilon, delta, p = .1, .01, 1
    norb, nelec = 6, (3,3)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    

    compatible = ucj_to_compatible(circuit)
    probs_compatible = [raw_estimate(circuit=compatible,
                                     outcome_state=b, 
                                     epsilon=epsilon, delta=delta, p=p) for b in bitstrings[:10]]
    assert_allclose(probs_compatible, exact_probs[:10], rtol=0, atol=0.05)

    compatible_fully_reduced = ucj_to_compatible_fully_reduced(circuit)
    probs_compatible_fully_reduced = [raw_estimate(circuit=compatible_fully_reduced, 
                                                   outcome_state=b, 
                                                   epsilon=epsilon, 
                                                   delta=delta, 
                                                   p=p) for b in bitstrings[:10]]
    assert_allclose(probs_compatible_fully_reduced, exact_probs[:10], rtol=0, atol=0.05)
