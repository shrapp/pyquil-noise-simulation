##############################################################################
# Written by Shahar Rapp and Ze'ev Binnes, added on Rigettis pyquil.
#
#    A new noise model based on pyquil.noise,
#    which works by adding noisy I gates after long operations.
##############################################################################

from typing import Dict, List, Sequence, Optional
import numpy as np
from pyquil.quilbase import Gate, DefGate
from pyquil.quil import Program
from pyquil.api import QuantumComputer
from pyquil.noise import NoiseModel, KrausModel, _get_program_gates, INFINITY
from pyquil.noise import damping_kraus_map, dephasing_kraus_map, combine_kraus_maps
from Calibrations import Calibrations, get_T_values, get_readout_fidelity

Noisy_I_1Q_name = "Noisy_I_1Q_gate"
Noisy_I_2Q_name = "Noisy_I_2Q_gate"

class Noise_types:
    """
    Contains the noise types we should add to the program.
    The user can choose what types of noise to add,
    by seting them as `True` or `False`.

    !!! when we add a model for the fidelity, we should default it to `True`.
    """
    def __init__(self) -> None:
        self.decoherence_2Q = True
        self.decoherence_1Q = False
        self.fidelity = False
        self.readout = True
    def set_noise(self, decoherence_2q, decoherence_1q, fidelity, readout):
        self.decoherence_2Q = decoherence_2q
        self.decoherence_1Q = decoherence_1q
        self.fidelity = fidelity
        self.readout = readout

def damping_after_dephasing(T1: float, T2: float, gate_time: float) -> List[np.ndarray]:
    """
    Generate the Kraus map corresponding to the composition
    of a dephasing channel followed by an amplitude damping channel.

    :param T1: The amplitude damping time
    :param T2: The dephasing time
    :param gate_time: The gate duration.
    :return: A list of Kraus operators.
    """
    assert T1 >= 0
    assert T2 >= 0

    if T1 != INFINITY:
        damping = damping_kraus_map(p=1 - np.exp(-float(gate_time) / float(T1)))
    else:
        damping = [np.eye(2)]
    if T2 != INFINITY:
        gamma_phi = float(gate_time) / float(T2)
        if T1 != INFINITY:
            if T2 > 2 * T1:
                T2 = 2 * T1     # this is what we changed
                gamma_phi = float(gate_time) / float(T2)
            gamma_phi -= float(gate_time) / float(2 * T1)
        dephasing = dephasing_kraus_map(p=0.5 * (1 - np.exp(-gamma_phi)))
    else:
        dephasing = [np.eye(2)]
    return combine_kraus_maps(damping, dephasing)

def create_noise_model(
    gates: Sequence[Gate],
    T1: Dict[int, float],
    T2: Dict[int, float],
    ro_fidelity: Dict[int, float],
    gate_time_1q: float = 40e-9,
    gate_time_2q: float = 180e-09,
    ) -> NoiseModel:
    """
    Create the instance of NoiseModel for our program.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param gate_time_1q: The duration of the one-qubit gates. By default, this is 40 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ. By default, this is 180 ns.
    :param ro_fidelity: The readout assignment fidelity dictionary indexed by qubit id.
        :math:`F = (p(0|0) + p(1|1))/2` 
    :return: A NoiseModel with the appropriate Kraus operators defined.
    """
    all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))

    noisy_identities_1q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_1q) for q in all_qubits
    }
    noisy_identities_2q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_2q) for q in all_qubits
    }
    kraus_maps = []
    for g in gates:
        targets = tuple(t.index for t in g.qubits)
        noisy_I = []
        if g.name == Noisy_I_1Q_name:
            noisy_I = noisy_identities_1q[targets[0]]
        elif g.name == Noisy_I_2Q_name:
            noisy_I = noisy_identities_2q[targets[0]]

        kraus_maps.append(
            KrausModel(
                g.name,
                tuple(g.params),
                targets,
                noisy_I,
                1.0, # will be fixed
            )
        )

    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1.0 - f_ro], [1.0 - f_ro, f_ro]])

    return NoiseModel(kraus_maps, aprobs)

def add_decoherence_noise_to_I(
    prog: Program,
    T1: Dict[int, float],
    T2: Dict[int, float],
    ro_fidelity: Dict[int, float],
    gate_time_1q: float = 40e-9,
    gate_time_2q: float = 180e-09,
) -> Program:
    """
    Applies the model on the different kindes of I.
    :param prog: The program including I's that are not noisy yet.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param ro_fidelity: The readout assignment fidelity dictionary indexed by qubit id.
    :param gate_time_1q: The duration of the one-qubit gates. By default, this is 40 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ. By default, this is 180 ns.
    :return: A new program with noisy operators.
    """
    # collect the noisy gates:
    gates = [i for i in _get_program_gates(prog) 
            if (i.name == Noisy_I_1Q_name) or (i.name == Noisy_I_2Q_name)]
    # define readout fidelity dict for all qubits in the program:
    ro_fidelity_prog_qubits = {q: ro_fidelity[q] for q in prog.get_qubits()}
    noise_model = create_noise_model(
        gates,
        T1=T1,
        T2=T2,
        ro_fidelity=ro_fidelity_prog_qubits,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
    )
    # add Kraus definition pragmas
    for k in noise_model.gates:
        prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)
    # add readout noise pragmas
    for q, ap in noise_model.assignment_probs.items():
        prog.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
    return prog

def add_noise_to_program(
    qc: QuantumComputer, 
    p: Program, 
    is_native: bool=False,
    calibrations: Optional[Calibrations] = None,
    noise_types: Noise_types = Noise_types()
    ):
    """
    Add generic damping and dephasing noise to a program.
    Noise is added to all qubits, after a 2-qubit gate operation.
    This function will define new I gates and add Kraus noise to these gates.
    :param qc: A Quantum computer object
    :param p: A pyquil program
    :param is_native: bool, should be `True` if the program is already in native pyquil
    :param calibrations: optional, can get the calibrations in advance, 
        instead of producing them from the URL.
    :param noise_types: can define what types of noise to add to the program.
        the options and defaults are listed in the class `Noise_types`.
    :return: A new program with noisy operators.
    """
    new_p = Program()
    if not is_native:
        p = qc.compiler.quil_to_native_quil(p)
    Noisy_I_2Q_gate = define_noisy_I_gates(new_p, noise_types)
    qubits = p.get_qubits()
    for i in p:
        new_p += i
        if isinstance(i, Gate):
            targets = tuple(t.index for t in i.qubits)
            # for 2-qubit gates, add decoherence noise for all qubits in qc
            if len(targets) == 2:
                if noise_types.fidelity:
                    for q in targets:
                        # new_p += Noisy_I_2Q_fidelity(q)
                        continue
                if noise_types.decoherence_2Q:
                    for q in qubits:
                        if q not in targets or not noise_types.fidelity:
                            new_p += Noisy_I_2Q_gate(q)
    qc_name = get_qc_name(qc)
    if calibrations != None:
        T1 = calibrations.T1
        T2 = calibrations.T2
        readout_fidelity = calibrations.readout
    else:
        T1, T2 = get_T_values(qc_name)
        readout_fidelity = get_readout_fidelity(qc_name)
    new_p = add_decoherence_noise_to_I(prog=new_p, T1=T1, T2=T2, ro_fidelity=readout_fidelity)
    new_p.wrap_in_numshots_loop(p.num_shots)    # wrap in original programs numshots
    return new_p

def get_qc_name(qc: QuantumComputer):
    """
    returns the name of the quantum computer `qc`, 
    without the ending 'qvm' if it exists.
    """
    name = qc.name
    if (name[-4:] == "-qvm"):
        name = name[0:-4]
        return name
    else:
        return name

def define_noisy_I_gates(p: Program, noise_types: Noise_types):
    """
    Adds a definition of the noisy-I gate to the program.

    Currenntly supports only Noisy-I after 2Q gate.

    :param p: a program.
    :param noise_types: the noise types to add to the program.
    :return: the new gate, that can be added now to the program.
    """
    I2QG_def = DefGate(Noisy_I_2Q_name, np.eye(2))
    Noisy_I_2Q_gate = I2QG_def.get_constructor()
    if noise_types.decoherence_2Q:
        p += I2QG_def
    return Noisy_I_2Q_gate