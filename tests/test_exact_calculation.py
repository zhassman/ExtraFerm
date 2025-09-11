import math

import pytest
from numpy.testing import assert_allclose

from extended_matchgate_simulator.exact import exact_calculation
from extended_matchgate_simulator.utils import (
    get_bitstrings_and_probs,
    make_parameterized_controlled_phase_circuit,
    ucj_to_compatible,
    ucj_to_compatible_fully_reduced,
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
def test_six_qubit_exact_calculation(mean, converter):
    var = 0.1
    norb, nelec = 3, (1, 1)
    circuit = make_parameterized_controlled_phase_circuit(
        norb, nelec, mean, var, reduced_interaction=False
    )
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    probs = exact_calculation(
        circuit=converter(circuit),
        outcome_states=bitstrings,
    )
    assert_allclose(probs, exact_probs)


@pytest.mark.parametrize("mean", [0, math.pi])
@pytest.mark.parametrize("converter", CONVERTERS)
def test_twelve_qubit_exact_calculation(mean, converter):
    var = 0.1
    norb, nelec = 6, (3, 3)
    circuit = make_parameterized_controlled_phase_circuit(
        norb, nelec, mean, var, reduced_interaction=False
    )
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    probs = exact_calculation(
        circuit=converter(circuit),
        outcome_states=bitstrings,
    )
    assert_allclose(probs, exact_probs)


@pytest.mark.parametrize("mean", [0, math.pi])
@pytest.mark.parametrize("converter", CONVERTERS)
def test_twenty_four_qubit_exact_calculation(mean, converter):
    var = 0.1
    norb, nelec = 12, (3, 3)
    circuit = make_parameterized_controlled_phase_circuit(
        norb, nelec, mean, var, reduced_interaction=False
    )
    bitstrings, exact_probs = get_bitstrings_and_probs(circuit, norb, nelec)
    probs = exact_calculation(
        circuit=converter(circuit),
        outcome_states=bitstrings,
    )
    assert_allclose(probs, exact_probs)
