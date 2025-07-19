def chemistry_to_compatible():
    """
    Takes an ffsim-style qiskit quantum circuit and makes it compatible 
    with the simulator.
    """
    # decompose hf to x gates
    # decompose diag coulomb to c-phase gates
    # remove global phase gate
