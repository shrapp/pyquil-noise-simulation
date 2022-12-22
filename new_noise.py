import sys
from typing import Dict, List, Set, Tuple, Union
import numpy as np
from pyquil.quil import Program
from pyquil.noise import _get_program_gates, damping_after_dephasing, INFINITY, NO_NOISE, tensor_kraus_maps, KrausModel, combine_kraus_maps, NoiseModel, _noise_model_program_header, NoisyGateUndefined
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES

def get_noisy_gate(gate: Gate) -> Tuple[np.ndarray, str]:
    """Given a gate ``Instruction``, turn it into a matrix.

    :param gate: the instruction
    :return: matrix.
    """
    if len(gate.params) > 0:
        return QUANTUM_GATES[gate.name](*gate.params), ("NOISY-" + gate.name + str(gate.params))
    else:
        return QUANTUM_GATES[gate.name],("NOISY-" + gate.name)



def add_noise_to_program(
    prog: "Program",
    T1: Union[Dict[int, float], float] = 3e-6,
    T2: Union[Dict[int, float], float] = 3e-6,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 150e-09,
    ro_fidelity: Union[Dict[int, float], float] = 0.95,
) -> "Program":
    """
    Add generic damping and dephasing noise to a program.

    This high-level function is provided as a convenience to investigate the effects of a
    generic noise model on a program. For more fine-grained control, please investigate
    the other methods available in the ``pyquil.noise`` module.

    The default noise parameters

    - T1 = 3 us
    - T2 = 3 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    This function will add Kraus noise to gates. 

    :param prog: A pyquil program
    :param T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 3 us.
    :param T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 3 us.
    :param gate_time_1q: The duration of the one-qubit gates.
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates.
        By default, this is 150 ns.
    :param ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noise.
    """
    gates = _get_program_gates(prog)
    
    all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))
    if isinstance(T1, dict):
        all_qubits.update(T1.keys())
    if isinstance(T2, dict):
        all_qubits.update(T2.keys())
    if isinstance(ro_fidelity, dict):
        all_qubits.update(ro_fidelity.keys())

    if not isinstance(T1, dict):
        T1 = {q: T1 for q in all_qubits}

    if not isinstance(T2, dict):
        T2 = {q: T2 for q in all_qubits}

    if not isinstance(ro_fidelity, dict):
        ro_fidelity = {q: ro_fidelity for q in all_qubits}

    noisy_identities_1q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_1q) for q in all_qubits
    }
    noisy_identities_2q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_2q) for q in all_qubits
    }
    kraus_maps = []
    for g in gates:
        targets = tuple(t.index for t in g.qubits)
        if g.name in NO_NOISE:
            continue
        matrix, _ = get_noisy_gate(g)

        if len(targets) == 1:
            noisy_I = noisy_identities_1q[targets[0]]
        else:
            if len(targets) != 2:
                raise ValueError("Noisy gates on more than 2Q not currently supported")

            # note this ordering of the tensor factors is necessary due to how the QVM orders
            # the wavefunction basis
            noisy_I = tensor_kraus_maps(noisy_identities_2q[targets[1]], noisy_identities_2q[targets[0]])
        kraus_maps.append(
            KrausModel(
                g.name,
                tuple(g.params),
                targets,
                combine_kraus_maps(noisy_I, [matrix]),
                # FIXME (Nik): compute actual avg gate fidelity for this simple
                # noise model
                1.0,
            )
        )
    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1.0 - f_ro], [1.0 - f_ro, f_ro]])

    noise_model = NoiseModel(kraus_maps, aprobs)



    p = Program()
    defgates: Set[str] = set()
    for k in noise_model.gates:

        # obtain ideal gate matrix and new, noisy name by looking it up in the NOISY_GATES dict
        try:
            ideal_gate, new_name = get_noisy_gate(k)

            # if ideal version of gate has not yet been DEFGATE'd, do this
            if new_name not in defgates:
                p.defgate(new_name, ideal_gate)
                defgates.add(new_name)
        except NoisyGateUndefined:
            print(
                "WARNING: Could not find ideal gate definition for gate {}".format(k.gate),
                file=sys.stderr,
            )
            new_name = k.gate

        # define noisy version of gate on specific targets
        p.define_noisy_gate(new_name, k.targets, k.kraus_ops)

    # define noisy readouts
    for q, ap in noise_model.assignment_probs.items():
        p.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
    new_prog = p



    for i in prog:
        if isinstance(i, Gate) and noise_model.gates:
            try:
                _, new_name = get_noisy_gate(i)
                new_prog += Gate(new_name, [], i.qubits)
            except NoisyGateUndefined:
                new_prog += i
        else:
            new_prog += i
    return prog.copy_everything_except_instructions() + new_prog