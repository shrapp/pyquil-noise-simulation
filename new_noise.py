##############################################################################
# Written by Shahar Rapp and Ze'ev Binnes, added on Rigetti's pyquil.
#
#    A new noise model based on pyquil.noise,
#    which works by adding "noisy I" gates after long operations.
##############################################################################

from typing import Dict, List, Sequence, Optional, Tuple, Any

import numpy as np
from pyquil.api import QuantumComputer
from pyquil.noise import NoiseModel, KrausModel, _get_program_gates, INFINITY, pauli_kraus_map, damping_kraus_map, \
	dephasing_kraus_map, combine_kraus_maps
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Gate, DefGate, Pragma, DelayQubits

from Calibrations import Calibrations

Noisy_I_1Q_name = "Noisy_I_gate_for_1Q"
Noisy_I_2Q_name = "Noisy_I_gate_for_2Q"


class NoiseTypes:
	"""
    Contains the noise types we should add to the program.
    The user can choose what types of noise to add,
    by setting them as `True` or `False`.

    Contains also the definitions add gate names of the noisy-I gates,
    under the field `gates`.

    !!! when we add a model for the fidelity, we should default it to `True`.
    """

	def __init__(self, decoherence_2q: bool = True, decoherence_1q: bool = False, fidelity: bool = False,
	             readout: bool = True) -> None:
		self.decoherence_2Q = decoherence_2q
		self.decoherence_1Q = decoherence_1q
		self.fidelity = fidelity
		self.readout = readout
		self.definitions = {}
		self.gates = {}
		self.define_noisy_gates()

	def define_noisy_gates(self) -> None:
		if self.decoherence_2Q:
			# future improvement may add implementation for different times of 2Q gates
			dg = DefGate(Noisy_I_2Q_name, np.eye(2))
			self.definitions[Noisy_I_2Q_name] = dg
			self.gates[Noisy_I_2Q_name] = dg.get_constructor()
		if self.decoherence_1Q:
			# future improvement may add implementation for different times of 1Q gates
			dg = DefGate(Noisy_I_1Q_name, np.eye(2))
			self.definitions[Noisy_I_1Q_name] = dg
			self.gates[Noisy_I_1Q_name] = dg.get_constructor()
		if self.fidelity:
			dg = DefGate("Depolarising_1Q_gate", np.eye(2))
			self.definitions["Depolarising_1Q_gate"] = dg
			self.gates["Depolarising_1Q_gate"] = dg.get_constructor()
			for name in ["Depolarising_CPHZE", "Depolarising_CZ", "Depolarising_XY"]:
				dg = DefGate(name, np.eye(4))
				self.definitions[name] = dg
				self.gates[name] = dg.get_constructor()


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
				T2 = 2 * T1  # this is what we changed
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
		gate_time: float,
) -> NoiseModel:
	"""
    Create the instance of NoiseModel for our program.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param gate_time: The duration of a gate
    :return: A NoiseModel with the appropriate Kraus operators defined.
    """
	all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))

	noisy_identities = {
		q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time) for q in all_qubits
	}
	kraus_maps = []
	for g in gates:
		targets = tuple(t.index for t in g.qubits)
		noisy_I = noisy_identities[targets[0]]

		kraus_maps.append(
			KrausModel(
				g.name,
				tuple(g.params),
				targets,
				noisy_I,
				1.0,  # will be fixed
			)
		)
	return NoiseModel(kraus_maps, {})


def add_decoherence_noise(
		prog: Program,
		T1: Dict[int, float],
		T2: Dict[int, float],
		gate_time_1q: float = 32e-9,
		gate_time_2q: float = 176e-09,
) -> Program:
	"""
    Applies the model on the different kinds of I.
    :param prog: The program including I's that are not noisy yet.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param gate_time_1q: The duration of the one-qubit gates. By default, this is 40 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ. By default, this is 180 ns.
    :return: A new program with noisy operators.
    """
	# collect the noisy gates:
	gates_2q = [i for i in _get_program_gates(prog) if i.name == Noisy_I_2Q_name]
	noise_model_2q = create_noise_model(
		gates_2q,
		T1=T1,
		T2=T2,
		# for future improvement: use the specific gate time if known
		gate_time=gate_time_2q,
	)
	gates_1q = [i for i in _get_program_gates(prog) if i.name == Noisy_I_1Q_name]
	noise_model_1q = create_noise_model(
		gates_1q,
		T1=T1,
		T2=T2,
		# for future improvement: use the specific gate time if known
		gate_time=gate_time_1q,
	)
	# add Kraus definition pragmas
	for k in noise_model_2q.gates:
		prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)
	for k in noise_model_1q.gates:
		prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)
	return prog


def add_readout_noise(
		prog: Program,
		ro_fidelity: Dict[int, float],
) -> Program:
	"""
    adds readout noise to the program.
    :param prog: The program without readout noise yet.
    :param ro_fidelity: The readout assignment fidelity dictionary indexed by qubit id.
    """
	# define readout fidelity dict for all qubits in the program:
	ro_fidelity_prog_qubits = {q: ro_fidelity[q] for q in prog.get_qubits()}
	# define a noise model with readout noise
	assignment_probs = {}
	for q, f_ro in ro_fidelity_prog_qubits.items():
		assignment_probs[q] = np.array([[f_ro, 1.0 - f_ro], [1.0 - f_ro, f_ro]])
	noise_model = NoiseModel([], assignment_probs)
	# add readout noise pragmas
	for q, ap in noise_model.assignment_probs.items():
		prog.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
	return prog

def create_depolarizing_noise_model(
		gates: Sequence[Gate],
		p: Dict,
		num_qubits: int
) -> NoiseModel:

	all_qubits = []
	for g in gates:
		qubits = [t.index for t in g.qubits]
		qubits.sort(key = lambda x: int(x))
		if qubits is None:
			qubits = []
		if len(qubits) > 1:
			all_qubits.append(str(qubits[0])+'-'+str(qubits[1]))
		elif len(qubits) == 1:
			all_qubits.append(qubits[0])
	all_qubits = set(all_qubits)
		
	# all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), [])) # should have also touples

	noisy_identities = {
		q: depolarizing_kraus(num_qubits, p.get(q, 1.0)) for q in all_qubits # 1 is wrong
	}
	kraus_maps = []
	for g in gates:
		targets = tuple(t.index for t in g.qubits)
		qq = targets
		if num_qubits > 1:
			qq = sorted(list(targets))
			qq = [str(qq[0])+'-'+str(qq[1])]
		noisy_I = noisy_identities[qq[0]]

		kraus_maps.append(
			KrausModel(
				g.name,
				tuple(g.params),
				targets,
				noisy_I,
				1.0,  # will be fixed
			)
		)
	return NoiseModel(kraus_maps, {})


def add_depolarizing_noise(prog: Program, p: Dict[str, Dict[int, float]]):

	gates = [i for i in _get_program_gates(prog) if i.name == "Depolarising_1Q_gate"]
	noise_model = create_depolarizing_noise_model(
		gates,
		p=p["Depolarising_1Q_gate"], 
		num_qubits=1
	)
	for k in noise_model.gates:
		prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)

	for name in ["Depolarising_CPHZE", "Depolarising_CZ", "Depolarising_XY"]:
		gates = [i for i in _get_program_gates(prog) if i.name == name]
		noise_model = create_depolarizing_noise_model(
			gates,
			p=p[name], 
			num_qubits=2
		)
		for k in noise_model.gates:
			prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)
	return prog


def replace_delay_with_noisy_I(p: Program) -> Tuple[Program, List[Dict[str, Any]]]:
	"""
    replace the instructions of type `DelayQubits` with `noisy-I` gates.

    :param p: a program. 

    :return: `new_p`: a converted program. 
    :return: `noisy_gates`: info about the gates that replace `DELAY`.
    """
	noisy_gates = []
	new_p = Program()
	new_p._defined_gates = p._defined_gates
	idx = 0
	for i in p:
		if not (isinstance(i, DelayQubits) or (isinstance(i, Pragma) and i.command == "DELAY")):
			new_p += i
			continue
		dg = DefGate("Noisy-DELAY-" + str(idx), np.eye(2))
		new_p += dg
		gate = dg.get_constructor()
		if isinstance(i, DelayQubits):
			duration = i.duration
			for q in i.qubits:
				new_p += gate(q)
		else:  # pragma
			duration = float(i.args[-1])
			for q in i.args[1:-1]:
				new_p += gate(Qubit(q))
		noisy_gates.append({"gate_name": "Noisy-DELAY-" + str(idx), "duration": duration})
		idx += 1
	return new_p, noisy_gates


def add_delay_maps(prog: Program, gate_info: Dict[str, Any], T1: Dict[int, float], T2: Dict[int, float]) -> None:
	"""
    Add kraus maps for a `DELAY` instruction, 
    that was converted already into `noisy-I` gate.

    :param prog: the program to add the maps to.
    :param gate_info: a Dictionary with the gates name and duration.
    :param T1: Dictionary with T1 times.
    :param T2: Dictionary with T2 times.
    """
	gates = [i for i in _get_program_gates(prog) if i.name == gate_info["gate_name"]]
	noise_model = create_noise_model(
		gates,
		T1=T1,
		T2=T2,
		gate_time=gate_info["duration"],
	)
	for k in noise_model.gates:
		prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)


def add_noise_to_program(
		qc: QuantumComputer,
		p: Program,
		convert_to_native: bool = True,
		calibrations: Optional[Calibrations] = None,
		noise_types: NoiseTypes = NoiseTypes()
) -> Program:
	"""
    Add generic damping and dephasing noise to a program.
    Noise is added to all qubits, after a 2-qubit gate operation.
    This function will define new "I" gates and add Kraus noise to these gates.
    :param qc: A Quantum computer object
    :param p: A pyquil program
    :param convert_to_native: bool, put `False` if the program is already in native pyquil or is not needed.
    :param calibrations: optional, can get the calibrations in advance, 
        instead of producing them from the URL.
    :param noise_types: can define what types of noise to add to the program.
        the options and defaults are listed in the class `NoiseTypes`.
    :return: A new program with noisy operators.
    """
	if convert_to_native:
		p = qc.compiler.quil_to_native_quil(p)
	qubits = p.get_qubits()

	new_p = Program()
	# TODO: add only the gates that are used in the program
	for definition in noise_types.definitions.values():
		new_p += definition
	for i in p:
		new_p += i
		if isinstance(i, Gate):
			targets = tuple(t.index for t in i.qubits)
			# for 2-qubit gates, add decoherence noise for all qubits in qc
			if len(targets) == 2:
				if noise_types.fidelity:
					new_p += noise_types.gates["Depolarising_" + i.name](targets[0], targets[1])
				if noise_types.decoherence_2Q:
					for q in qubits:
						if q not in targets or not noise_types.fidelity:
							new_p += noise_types.gates[Noisy_I_2Q_name](q)
			elif len(targets) == 1:
				if noise_types.fidelity:
					for q in targets:
						new_p += noise_types.gates["Depolarising_1Q_gate"](targets[0])
				if noise_types.decoherence_1Q:
					for q in qubits:
						if q not in targets or not noise_types.fidelity:
							new_p += noise_types.gates[Noisy_I_1Q_name](q)

	# deal with delay:
	new_p, delay_gates = replace_delay_with_noisy_I(new_p)

	if calibrations is None:
		calibrations = Calibrations(qc=qc)

	if noise_types.fidelity:

		new_p = add_depolarizing_noise(prog=new_p, p={
			"Depolarising_1Q_gate": calibrations.fidelity_1q, 
			"Depolarising_CPHZE": calibrations.fidelity_CPHASE, 
			"Depolarising_CZ": calibrations.fidelity_CZ,
			"Depolarising_XY": calibrations.fidelity_XY})

	# add kraus maps declarations:
	if noise_types.decoherence_1Q or noise_types.decoherence_2Q:
		new_p = add_decoherence_noise(prog=new_p, T1=calibrations.T1, T2=calibrations.T2)
	if noise_types.readout:
		new_p = add_readout_noise(prog=new_p, ro_fidelity=calibrations.readout)

	for gate in delay_gates:
		add_delay_maps(new_p, gate, calibrations.T1, calibrations.T2)

	new_p.wrap_in_numshots_loop(p.num_shots)  # wrap in original program's numshots
	return new_p


def depolarizing_kraus(num_qubits: int, p: float = .95) -> List[np.ndarray]:
	"""
    Generate the Kraus operators corresponding to a given unitary
    single qubit gate followed by a depolarizing noise channel.

    :params float num_qubits: either 1 or 2 qubit channel supported
    :params float p: parameter in depolarizing channel as defined by: p $\rho$ + (1-p)/d I
    :return: A list, eg. [k0, k1, k2, k3], of the Kraus operators that parametrize the map.
    :rtype: list
    """
	num_of_operators = 4 ** num_qubits
	probabilities = [p + (1.0 - p) / num_of_operators]
	probabilities += [(1.0 - p) / num_of_operators] * (num_of_operators - 1)
	return pauli_kraus_map(probabilities)


