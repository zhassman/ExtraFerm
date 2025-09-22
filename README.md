# Extended Matchgate Simulator

## Overview

This library can be used to estimate the Born-rule probability of measurement outcomes produced by particle-number conserving **extended matchgate circuits**:

- **Extended matchgate**: the universal gate set consisting of matchgates + controlled‑phase gates  
- **Particle-number conserving**: the Hamming weight of initial state is preserved at all points by the circuit

## Circuit Extent

Our simulator runs most efficiently when the circuit’s **extent** is small. The extent is defined as:

$$
\xi^* = \prod_{j=1}^k \bigl(\cos(\theta_j/4) + \sin(\theta_j/4)\bigr)^2
$$

where $\theta_j$ is the angle of the $j$-th controlled-phase gate.

## Use

Users should interact with the simulator through interface.py

## Algorithms

- **`raw_estimate()`** considers additive error $\epsilon$, failure probability $\delta$, and Born-rule probability upper bound $p$. This algorithm is best for general-purpose estimation of probabilities for moderate values of $\epsilon, \delta$.

- **`estimate()`** considers additive error $\epsilon$ and failure probability $\delta$ and is best for estimating outcome measurements with very high accuracy (small $\epsilon, \delta$). This routine enjoys further speedups when the probability that we wish to estimate is small.

- **`exact_calculation()`** computes exact probabilities and is useful when the number of controlled-phase gates is very small (<15).

## Setup

Install this repository

```bash
git clone https://github.com/zhassman/Extended-Matchgate-Simulator.git
cd Extended-Matchgate-Simulator
pip install -e .
```
