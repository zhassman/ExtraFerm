import numpy as np
from numpy.testing import assert_allclose

from extraferm.estimation import estimate
from extraferm.interface import outcome_probabilities
from extraferm.raw_estimation import raw_estimate, raw_estimate_lucj
from extraferm.utils import ucj_to_compatible
from extraferm.utils_dev import (
    get_bitstrings_and_probs,
    make_parameterized_controlled_phase_circuit,
)


def test_raw_estimate_batch_reproducibility():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=42,
    )
    probs2 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=42,
    )

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_raw_estimate_single_reproducibility():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = [
        raw_estimate(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            p=p,
            seed=42,
        )
        for b in bitstrings
    ]
    probs2 = [
        raw_estimate(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            p=p,
            seed=42,
        )
        for b in bitstrings
    ]

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_raw_estimate_lucj_batch_reproducibility():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = raw_estimate_lucj(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=42,
    )
    probs2 = raw_estimate_lucj(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=42,
    )

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_raw_estimate_lucj_single_reproducibility():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = [
        raw_estimate_lucj(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            p=p,
            seed=42,
        )
        for b in bitstrings
    ]
    probs2 = [
        raw_estimate_lucj(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            p=p,
            seed=42,
        )
        for b in bitstrings
    ]

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_raw_estimate_reuse_reproducibility():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        reuse_trajectories=True,
        seed=42,
    )
    probs2 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        reuse_trajectories=True,
        seed=42,
    )

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_estimate_batch_reproducibility():
    var = 0.1
    epsilon, delta = 0.1, 0.01
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        seed=42,
    )
    probs2 = estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        seed=42,
    )

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_estimate_single_reproducibility():
    var = 0.1
    epsilon, delta = 0.1, 0.01
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = [
        estimate(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            seed=42,
        )
        for b in bitstrings
    ]
    probs2 = [
        estimate(
            circuit=ucj_to_compatible(circuit),
            outcome_states=b,
            epsilon=epsilon,
            delta=delta,
            seed=42,
        )
        for b in bitstrings
    ]

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_outcome_probabilities_reproducibility():
    var = 0.1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = outcome_probabilities(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        trajectory_count=1000,
        seed=42,
    )
    probs2 = outcome_probabilities(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        trajectory_count=1000,
        seed=42,
    )

    assert_allclose(probs1, probs2, rtol=0, atol=1e-12)


def test_different_seeds_produce_different_results():
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    mean = 0
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, _ = get_bitstrings_and_probs(circuit, norb, nelec)

    probs1 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=42,
    )
    probs2 = raw_estimate(
        circuit=ucj_to_compatible(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        seed=123,
    )

    max_diff = np.max(np.abs(probs1 - probs2))
    assert max_diff > 1e-10, (
        f"Results with different seeds are too similar: max_diff = {max_diff}"
    )
