import math

import pytest
from numpy.testing import assert_allclose

from extraferm.raw_estimation import raw_estimate
from extraferm.utils import (
    ucj_to_compatible,
    ucj_to_compatible_fully_reduced,
)
from extraferm.utils_dev import (
    get_bitstrings_and_probs,
    make_parameterized_controlled_phase_circuit,
)

MEANS = [
    0,
    math.pi / 4,
    math.pi / 2,
    3 * math.pi / 4,
    math.pi,
    -math.pi,
    -3 * math.pi / 4,
    -math.pi / 2,
    -math.pi / 4,
]

CONVERTERS = [
    ucj_to_compatible,
    ucj_to_compatible_fully_reduced,
]


@pytest.mark.parametrize("mean", MEANS)
@pytest.mark.parametrize("converter", CONVERTERS)
def test_six_qubit_raw_estimate_with_reuse(mean, converter):
    var = 0.1
    epsilon, delta, p = 0.1, 0.01, 1
    norb, nelec = 3, (1, 1)
    circuit = make_parameterized_controlled_phase_circuit(norb, nelec, mean, var)
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    probs = raw_estimate(
        circuit=converter(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        reuse_trajectories=True,
    )
    assert_allclose(probs, exact_probs, rtol=0, atol=0.05)


@pytest.mark.parametrize(
    "norb, nelec, mean, var, sample_size, reduced_interaction",
    [
        (6, (3, 3), 0, 0.1, 10, False),
        (63, (1, 1), 0, 1e-5, 5, True),
    ],
)
@pytest.mark.parametrize("converter", CONVERTERS)
def test_large_circuit_raw_estimate_with_reuse(
    norb,
    nelec,
    mean,
    var,
    sample_size,
    reduced_interaction,
    converter,
):
    epsilon, delta, p = 0.1, 0.01, 1
    circuit = make_parameterized_controlled_phase_circuit(
        norb, nelec, mean, var, reduced_interaction=reduced_interaction
    )
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    bitstrings = bitstrings[:sample_size]
    exact_probs = exact_probs[:sample_size]
    probs = raw_estimate(
        circuit=converter(circuit),
        outcome_states=bitstrings,
        epsilon=epsilon,
        delta=delta,
        p=p,
        reuse_trajectories=True,
    )
    assert_allclose(probs, exact_probs, rtol=0, atol=0.05)
