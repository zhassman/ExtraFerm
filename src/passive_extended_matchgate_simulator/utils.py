from scipy.linalg import block_diag
import numpy as np
import math
from typing import Tuple
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit


CircuitData = Tuple[
    float,                   # extent
    int,                     # negative_mask
    np.ndarray,              # normalized_angles, shape (C,)
    int,                     # initial_state
    np.ndarray,              # gate_types, shape (N,)
    np.ndarray,              # params,     shape (N,2)
    np.ndarray,              # qubits,     shape (N,2)
    np.ndarray,              # orb_indices,shape (N,)
    np.ndarray,              # orb_mats,   shape (M,D,D)
]


def extract_circuit_data(
    circuit: QuantumCircuit
) -> CircuitData:
    """
    Retrieves all the data needed for the Rust backend in one pass.

    Returns:
        extent (float)
            the extent of the circuit
        
        negative_mask (int): 
            bitmask of which conrolled phase angles were negative

        normalized_angles (NDArray[np.float64], shape (C,)): 
            array of controlled phase angles normalized to (–π,π]
        
        initial_state: bitmask from X gates

        initial_state (int)
            Bitmask of the initial X gates (qubit i is 1<<i).

        gate_types (ndarray[uint8], shape (N,))
            Numeric codes for each gate in the circuit.

        params (ndarray[float64], shape (N,2))
            For each gate k:
              params[k,0] = primary angle (θ)
              params[k,1] = secondary angle (β), or 0.0 if unused

        qubits (ndarray[uint64], shape (N,2))
            For each gate k:
              qubits[k,0] = target qubit index
              qubits[k,1] = control qubit index (or same as [k,0] for 1-qubit gates)

        orb_indices (ndarray[int64], shape (N,))
            For each gate k:
              -1 if not an orb_rot_jw,
              otherwise the index into orb_mats.

        orb_mats (ndarray[complex128], shape (M, D, D))
            Stack of M full block-diagonal JW rotation matrices,
            each of dimension DxD (where D = A.shape[0] + B.shape[0]).
    """
    controlled_phase_angles = []
    initial_state = 0
    seen_non_x   = False

    gate_types  = []
    params      = []
    qubits      = []
    orb_indices = []
    orb_mats    = []

    for instr in circuit.data:
        name = instr.operation.name

        if name == "x":
            if seen_non_x:
                raise ValueError(
                    "All X gates must appear consecutively at the beginning of the circuit."
                )
            q = instr.qubits[0]._index
            initial_state |= 1 << q
            continue

        seen_non_x = True

        if name == "cp":
            gate_types.append(1)
            theta = instr.operation.params[0]
            params.append([theta, 0.0])
            q1, q2 = instr.qubits[0]._index, instr.qubits[1]._index
            qubits.append([q1, q2])
            orb_indices.append(-1)
            controlled_phase_angles.append(theta)

        elif name == "xx_plus_yy":
            gate_types.append(2)
            theta, beta = instr.operation.params
            params.append([theta, beta])
            q1, q2 = instr.qubits[0]._index, instr.qubits[1]._index
            qubits.append([q1, q2])
            orb_indices.append(-1)

        elif name == "p":
            gate_types.append(3)
            theta = instr.operation.params[0]
            params.append([theta, 0.0])
            q = instr.qubits[0]._index
            qubits.append([q, q])
            orb_indices.append(-1)

        elif name == "orb_rot_jw":
            gate_types.append(4)
            params.append([0.0, 0.0])
            qubits.append([0, 0])
            orb_indices.append(len(orb_mats))
            A = np.asarray(instr.operation.orbital_rotation_a,
                           dtype=np.complex128)
            B = np.asarray(instr.operation.orbital_rotation_b,
                           dtype=np.complex128)
            M = block_diag(A, B)
            orb_mats.append(M)

        else:
            raise ValueError(f"Unexpected gate '{name}' in circuit.")
    
    if orb_mats:
        orb_mats_arr = np.stack(orb_mats, axis=0).astype(np.complex128)
    else:
        orb_mats_arr = np.zeros((0, 0, 0), dtype=np.complex128)

    normalized_angles = []
    negative_mask = 0

    for i, theta in enumerate(controlled_phase_angles):
        n = ((theta + math.pi) % (2 * math.pi)) - math.pi
        normalized_angles.append(n)
        if n < 0:
            negative_mask |= 1 << i

    extent = calculate_extent(normalized_angles)

    return (
        extent,
        negative_mask,
        np.array(normalized_angles, dtype=np.float64),
        initial_state,
        np.array(gate_types,  dtype=np.uint8),
        np.array(params,      dtype=np.float64),
        np.array(qubits,      dtype=np.uint64),
        np.array(orb_indices, dtype=np.int64),
        orb_mats_arr,
    )


def calculate_trajectory_count(
    epsilon: float,
    delta: float,
    extent: float,
    p: float,
) -> int:
    """
    Computes a lower bound on the number of trajectories needed for 
    the desired additive error and failure probability given the upper bound
    on outcome measurement probability and the extent of the circuit.
    
    Args:
        epsilon: additive error
        delta: failure probability
        extent: the extent of the circuit
        p: the Born rule probability upper bound
    """
    root_e = math.sqrt(extent)
    root_p = math.sqrt(p)
    numerator = (root_e + root_p) ** 2
    log_term = math.log(2 * math.exp(2) / delta)
    denominator = (math.sqrt(p + epsilon) - root_p) ** 2
    return math.ceil(2 * numerator * log_term / denominator)


def calculate_extent(
    normalized_angles: NDArray[np.float64]
) -> float:
    """
    Calculates the extent of the circuit given its controlled phase
    angles.

    Args:
        normalized_angles: normalized controlled phase angles, shape (C,)
    """
    prod = 1.0
    for theta in normalized_angles:
        t = abs(theta)
        prod *= (math.cos(t / 4.0) + math.sin(t / 4.0)) ** 2
    return prod


def chemistry_to_compatible():
    """
    Takes an ffsim-style qiskit quantum circuit and makes it compatible 
    with the simulator.
    """
    # decompose hf to x gates
    # decompose diag coulomb to c-phase gates
    # remove global phase gate
