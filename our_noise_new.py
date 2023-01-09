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

from our_noise import _get_program_gates, _decoherence_noise_model, _noise_model_program_header, get_noisy_gate, NoisyGateUndefined, NoiseModel

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
    new_prog = _noise_model_program_header(noise_model)
    for i in prog:
        if isinstance(i, Gate) and noise_model.gates:
            try:
                if i.name == 'I':
                    _, new_name = get_noisy_gate(i.name, tuple(i.params))
                    new_prog += Gate(new_name, [], i.qubits)
                else:
                    new_prog += i
            except NoisyGateUndefined:
                new_prog += i
        else:
            new_prog += i
    return prog.copy_everything_except_instructions() + new_prog

def add_decoherence_noise_to_I(
    prog: "Program",
    T1: Union[Dict[int, float], float] = 30e-6,
    T2: Union[Dict[int, float], float] = 30e-6,
    double_gate_time: bool = False, # maybe not needed
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
    if double_gate_time:
        gate_time_1q = 150e-9 # should be 180, but that had sqrt bugs.
    gates = [i for i in _get_program_gates(prog) if i.name == "I"]
    noise_model = _decoherence_noise_model(
        gates,
        T1=T1,
        T2=T2,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
        ro_fidelity=ro_fidelity,
    )
    return apply_noise_model_I(prog, noise_model)


def add_noise_to_program(qc, p, cal, recompile: bool=True):
    """
    Add generic damping and dephasing noise to a program.

    Noise is added to all qubits, after a 2-qubit gate operation.
    This function will define new I gates and add Kraus noise to these gates.

    :param qc: A Quantum computer object
    :param p: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param cal: A Calibration object: has calibration data for T1 and T2
    :param recompile: bool, should be False if the program is already compiled

    :return: A new program with noisy operators.
    """
    new_p = Program()
    # TODO: check if change to "to naitive"
    if recompile:
        p = qc.compile(p)
    for i in p:
        new_p += i
        if isinstance(i, Gate):
            targets = tuple(t.index for t in i.qubits)
            # for 2-qubit gates, add decoherence noise for all qubits in qc
            if len(targets) == 2:
                for q in qc.qubits():
                    new_p += I(q)
    new_p = add_decoherence_noise_to_I(new_p, cal.T1, cal.T2, True)
    return new_p
