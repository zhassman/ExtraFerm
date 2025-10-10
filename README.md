# ExtraFerm

## Overview

This library can be used to estimate the Born-rule probability of measurement outcomes produced by particle-number conserving **extended matchgate circuits**:

- **Extended matchgate**: the universal gate set consisting of matchgates + controlled‑phase gates  
- **Particle-number conserving**: the Hamming weight of initial state is preserved at all points by the circuit

## Circuit Extent

Our simulator runs most efficiently when the circuit’s **extent** is close to one. The extent is defined as:

$$
\xi^* = \prod_{j=1}^k \bigl(\cos(\theta_j/4) + \sin(\theta_j/4)\bigr)^2
$$

where $\theta_j$ is the angle of the $j$-th controlled-phase gate.

## Use

Users should interact with the simulator through the [`outcome_probabilities`](python/extended_matchgate_simulator/interface.py) function in `interface.py`. 

## Example

```python
probabilities = outcome_probabilities(
    circuit=my_circuit,
    outcome_states=bitstrings,
    additive_error=0.01,
    failure_probability=1e-3,
)
```

## Setup

Install this repository

```bash
git clone https://github.com/zhassman/Extended-Matchgate-Simulator.git
cd Extended-Matchgate-Simulator
pip install -e .
```
