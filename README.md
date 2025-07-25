# Passive Extended Matchgate Simulator

## Overview

This library can be used to estimate the Born-rule probability of measurement outcomes produced by **passive extended matchgate circuits**:

- **Extended matchgate**: the universal gate set consisting of matchgates + controlled‑phase gates  
- **Passive**: particle‑number conserving

## Circuit Extent

Our simulator runs most efficiently when the circuit’s **extent** is small. The extent is defined as:

$$
\xi^* = \prod_{j=1}^k \bigl(\cos(\theta_j/4) + \sin(\theta_j/4)\bigr)^2
$$

where $\theta_j$ is the angle of the $j$-th controlled-phase gate.

## Algorithms

- **`raw_estimate()`** considers additive error $\epsilon$, failure probability $\delta$, and Born-rule probability upper bound $p$. This algorithm is best for general-purpose estimation of probabilities for moderate values of $\epsilon, \delta$.

- **`estimate()`** considers additive error $\epsilon$ and failure probability $\delta$ and is best for estimating outcome measurements with very high accuracy (small $\epsilon, \delta$). This routine enjoys further speedups when the probability that we wish to estimate is small.

- **`exact_calculation()`** computes exact probabilities and is useful when the number of controlled-phase gates is very small (<15).

## Setup

Install the latest version of **ffsim**:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
pip install -e .
```

Install our simulator

```bash
git clone https://github.com/zhassman/Passive-Extended-Matchgate-Simulator.git
cd Passive-Extended-Matchgate-Simulator
pip install -e .
```
