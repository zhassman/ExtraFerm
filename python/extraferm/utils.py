import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from scipy.linalg import block_diag


class CircuitData(NamedTuple):
    """Circuit data structure for the Rust backend."""

    num_qubits: int
    extent: float
    negative_mask: int
    normalized_angles: NDArray[np.float64]
    initial_state: int
    gate_types: NDArray[np.uint8]
    params: NDArray[np.float64]
    qubits: NDArray[np.uint64]
    orb_indices: NDArray[np.int64]
    orb_mats: NDArray[np.complex128]


def extract_circuit_data(circuit: QuantumCircuit) -> CircuitData:
    """
    Retrieves all the data needed for the Rust backend in one pass.

    Returns:

        num_qubits (int)
            the number of qubits in the circuit

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
    num_qubits = circuit.num_qubits
    controlled_phase_angles = []
    initial_state = 0
    seen_non_x = False

    gate_types = []
    params = []
    qubits = []
    orb_indices = []
    orb_mats = []

    for instr in circuit.data:
        name = instr.operation.name

        if name == "x":
            if seen_non_x:
                raise ValueError(
                    "All X gates must appear consecutively at the beginning of the "
                    "circuit."
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
            a = np.asarray(instr.operation.orbital_rotation_a, dtype=np.complex128)
            b = np.asarray(instr.operation.orbital_rotation_b, dtype=np.complex128)
            m = block_diag(a, b)
            orb_mats.append(m)

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

    return CircuitData(
        num_qubits=num_qubits,
        extent=extent,
        negative_mask=negative_mask,
        normalized_angles=np.array(normalized_angles, dtype=np.float64),
        initial_state=initial_state,
        gate_types=np.array(gate_types, dtype=np.uint8),
        params=np.array(params, dtype=np.float64),
        qubits=np.array(qubits, dtype=np.uint64),
        orb_indices=np.array(orb_indices, dtype=np.int64),
        orb_mats=orb_mats_arr,
    )


def calculate_trajectory_count(
    epsilon: float,
    delta: float,
    p: float,
    extent: float,
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
    log_term = math.log(2.0) + 2.0 - math.log(delta)
    denom_temp = math.sqrt(p + epsilon) + root_p
    denominator = (epsilon**2) / (denom_temp**2)
    return math.ceil(2.0 * numerator * log_term / denominator)


def calculate_extent(normalized_angles: NDArray[np.float64]) -> float:
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


def ucj_to_compatible(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Takes an ffsim-style qiskit quantum circuit and makes it compatible
    with the simulator.

    Args:
        circuit: This should be a quantum circuit with only a hartree_fock_jw gate
        and a ucj_balanced_jw gate.
    """
    assert len(circuit.data) == 2
    assert circuit.data[0].operation.name == "hartree_fock_jw"
    assert circuit.data[1].operation.name == "ucj_balanced_jw"
    decomposed = circuit.decompose().decompose(
        gates_to_decompose=["slater_jw", "diag_coulomb_jw"]
    )

    num_qubits = decomposed.num_qubits
    compatible = QuantumCircuit(num_qubits)

    for instruction in decomposed.data:
        if instruction.operation.name == "x":
            q = instruction.qubits[0]._index
            compatible.append(instruction.operation, [q])

    for instruction in decomposed.data:
        if instruction.operation.name == "cp":
            q1 = instruction.qubits[0]._index
            q2 = instruction.qubits[1]._index
            compatible.append(instruction.operation, [q1, q2])

        elif instruction.operation.name == "orb_rot_jw":
            compatible.append(instruction.operation, range(num_qubits))

        elif instruction.operation.name == "global_phase":
            pass

    return compatible


def ucj_to_compatible_fully_reduced(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Takes an ffsim-style qiskit quantum circuit and makes it compatible
    with the simulator. Reduces orb_rot_jw all the way down to 2 qubit gates.

    Args:
        circuit: This should be a quantum circuit with only a hartree_fock_jw gate
        and a ucj_balanced_jw gate.
    """
    assert len(circuit.data) == 2
    assert circuit.data[0].operation.name == "hartree_fock_jw"
    assert circuit.data[1].operation.name == "ucj_balanced_jw"
    decomposed = circuit.decompose(reps=2)

    num_qubits = decomposed.num_qubits
    compatible = QuantumCircuit(num_qubits)

    for instruction in decomposed.data:
        if instruction.operation.name == "x":
            q = instruction.qubits[0]._index
            compatible.append(instruction.operation, [q])

    for instruction in decomposed.data:
        if instruction.operation.name in ["cp", "xx_plus_yy"]:
            q1 = instruction.qubits[0]._index
            q2 = instruction.qubits[1]._index
            compatible.append(instruction.operation, [q1, q2])

        elif instruction.operation.name == "p":
            q = instruction.qubits[0]._index
            compatible.append(instruction.operation, [q])

        elif instruction.operation.name == "global_phase":
            pass

    return compatible


def is_lucj(circuit: QuantumCircuit) -> bool:
    """
    Check if 'circuit' has the form:

      X* , orb_rot_jw , CP* , orb_rot_jw

    (ignoring any global_phase gates).
    """
    ops = [
        instr.operation.name for instr in circuit.data if instr.name != "global_phase"
    ]

    if ops.count("orb_rot_jw") != 2:
        return False

    first_idx = ops.index("orb_rot_jw")
    second_idx = ops.index("orb_rot_jw", first_idx + 1)

    if any(op != "x" for op in ops[:first_idx]):
        return False

    if any(op != "cp" for op in ops[first_idx + 1 : second_idx]):
        return False

    if second_idx != len(ops) - 1:
        return False

    return True
