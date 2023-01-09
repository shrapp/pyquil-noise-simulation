import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast

import numpy as np

from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare


from pyquil.quil import Program
from pyquil.api import QuantumComputer as PyquilApiQuantumComputer

from our_noise import _get_program_gates, NoiseModel, damping_after_dephasing, INFINITY, KrausModel
from get_t_value import get_T_values


def _noise_model_program_header_I(noise_model: NoiseModel) -> "Program":
    """
    Generate the header for a pyquil Program that uses ``noise_model`` to overload noisy gates.
    The program header consists of 3 sections:

        - The ``DEFGATE`` statements that define the meaning of the newly introduced "noisy" gate
          names.
        - The ``PRAGMA ADD-KRAUS`` statements to overload these noisy gates on specific qubit
          targets with their noisy implementation.
        - THe ``PRAGMA READOUT-POVM`` statements that define the noisy readout per qubit.

    :param noise_model: The assumed noise model.
    :return: A quil Program with the noise pragmas.
    """
    p = Program()
    defgates: Set[str] = set()
    for k in noise_model.gates:
        new_name = "I"
        if new_name not in defgates:
            p.defgate(new_name, np.eye(2))
            defgates.add(new_name)
        # define noisy version of gate on specific targets
        p.define_noisy_gate(new_name, k.targets, k.kraus_ops)

    return p

def apply_noise_model_I(prog: "Program", noise_model: NoiseModel) -> "Program":
    """
    Apply a noise model to a program and generate a 'noisy-fied' version of the program.
    this function adds noise only to I gates.
    :param prog: A Quil Program object.
    :param noise_model: A NoiseModel, either generated from an ISA or
        from a simple decoherence model.
    :return: A new program translated to a noisy gateset and with noisy readout as described by the
        noisemodel.
    """
    new_prog = _noise_model_program_header_I(noise_model)
    for i in prog:
        new_prog += i
    return prog.copy_everything_except_instructions() + new_prog

def _decoherence_noise_model_I(
    gates: Sequence[Gate],
    T1: Dict[int, float],
    T2: Dict[int, float],
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 150e-09,
    # ro_fidelity: Union[Dict[int, float], float] = 0.95,
) -> NoiseModel:
    """
    The default noise parameters

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
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
        noisy_I = noisy_identities_2q[targets[0]]

        kraus_maps.append(
            KrausModel(
                g.name,
                tuple(g.params),
                targets,
                noisy_I,
                1.0, # not needed
            )
        )
    aprobs = {}

    return NoiseModel(kraus_maps, aprobs)

def add_decoherence_noise_to_I(
    prog: "Program",
    T1: Dict[int, float],
    T2: Dict[int, float],
    gate_time_1q: float = 40e-9,
    gate_time_2q: float = 180e-09,
    ro_fidelity: Union[Dict[int, float], float] = 0.95,
) -> "Program":
    """
    Like add_decoherence_noise, but applies the model only on I
    :param double_gate_time: if True, the time for single I will be 180 ns,
    like a 2-qubit gate time.
    :return: A new program with noisy operators.
    """
    gates = [i for i in _get_program_gates(prog) if i.name == "I"] # change!
    noise_model = _decoherence_noise_model_I(
        gates,
        T1=T1,
        T2=T2,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
    )
    return apply_noise_model_I(prog, noise_model)


def add_noise_to_program(qc, p, recompile: bool=True):
    """
    Add generic damping and dephasing noise to a program.
    Noise is added to all qubits, after a 2-qubit gate operation.
    This function will define new I gates and add Kraus noise to these gates.
    :param qc: A Quantum computer object
    :param p: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param recompile: bool, should be False if the program is already compiled
    :return: A new program with noisy operators.
    """
    new_p = Program()
    if recompile:
        p = qc.compile(p)
    qubits = p.get_qubits()
    for i in p:
        new_p += i
        if isinstance(i, Gate):
            targets = tuple(t.index for t in i.qubits)
            # for 2-qubit gates, add decoherence noise for all qubits in qc
            if len(targets) == 2:
                for q in qubits:
                    new_p += I(q)
    qc_name = get_qc_name(qc)
    T1, T2 = get_T_values(qc_name)
    new_p = add_decoherence_noise_to_I(new_p, T1, T2, 180e-09)
    new_p.wrap_in_numshots_loop(p.num_shots)
    return new_p

def get_qc_name(qc):
    name = qc.name
    if (name[-4:] == "-qvm"):
        name = name[0:-4]
        return name
    else:
        return name
