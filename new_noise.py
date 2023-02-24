##############################################################################
# Written by Shahar Rapp and Ze'ev Binnes, added on Rigetti's pyquil.
#
#    A new noise model based on pyquil.noise,
#    which works by adding "noisy I" gates after long operations.
##############################################################################

from typing import Dict, List, Sequence, Optional, Tuple, Any
import json
import requests
import numpy as np

from pyquil.api import QuantumComputer
from pyquil.noise import NoiseModel, KrausModel, _get_program_gates, INFINITY, pauli_kraus_map, damping_kraus_map, \
	dephasing_kraus_map, combine_kraus_maps
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Gate, DefGate, Pragma, DelayQubits

Noisy_I_1Q_name = "Noisy_I_gate_for_1Q"
Noisy_I_2Q_name = "Noisy_I_gate_for_2Q"
Depolarizing_1Q_gate = "Depolarizing_1Q_gate"
Depolarizing_CPHASE = "Depolarizing_CPHASE"
Depolarizing_CZ = "Depolarizing_CZ"
Depolarizing_XY = "Depolarizing_XY"
No_Depolarizing = "No_Depolarizing"


class Calibrations:
	"""
    encapsulate the calibration data for Aspen-M-2 or Aspen-M-3 machine.

	contains:
		T1
		T2
		fidelity_1q
		readout
		fidelity_CPHASE
		fidelity_CZ
		fidelity_XY

    args: qc (QuantumComputer, optional): a Quantum Computer (Aspen-M-2 or Aspen-M-3). 
    Defaults to None, where the user can define his own calibration data.

	Notice: this class heavily relies on the specific way on which the lattices are written.
	this may change in time, and require changes in the class.
    """

	def __init__(self, qc: Optional[QuantumComputer] = None) -> None:
		if qc is None:
			return  # user can set his own values
		else:
			qc_name = self.get_qc_name(qc)
		if qc_name not in ["Aspen-M-2", "Aspen-M-3"]:
			raise ValueError("qc must be Aspen-M-2 or Aspen-M-3")
		else:
			url = "https://forest-server.qcs.rigetti.com/lattices/"
			response = requests.get(url + qc_name)
			file = json.loads(response.text)
			self.calibrations = file["lattice"]["specs"]
			self.create_1q_dicts()
			self.create_2q_dicts()

	def create_1q_dicts(self):
		qs = self.calibrations['1Q'].keys()
		t1 = [self.calibrations['1Q'][q]['T1'] for q in qs]
		t2 = [self.calibrations['1Q'][q]['T2'] for q in qs]
		fidelities = [self.calibrations['1Q'][q]["f1QRB"] for q in qs]
		readout = [self.calibrations['1Q'][q]["fRO"] for q in qs]
		qubits_indexes = [int(q) for q in qs]
		self.T1 = dict(zip(qubits_indexes, t1))
		self.T2 = dict(zip(qubits_indexes, t2))
		self.fidelity_1q = dict(zip(qs, fidelities))
		self.readout_fidelity = dict(zip(qubits_indexes, readout))

	def create_2q_dicts(self):
		pairs = self.calibrations['2Q'].keys()
		cphase = [self.calibrations['2Q'][pair].get("fCPHASE", 1.0) for pair in pairs]
		self.fidelity_CPHASE = dict(zip(pairs, cphase))
		cz = [self.calibrations['2Q'][pair].get("fCZ", 1.0) for pair in pairs]
		self.fidelity_CZ = dict(zip(pairs, cz))
		xy = [self.calibrations['2Q'][pair].get("fXY", 1.0) for pair in pairs]
		self.fidelity_XY = dict(zip(pairs, xy))

	def get_qc_name(self, qc: QuantumComputer):
		"""
		returns the name of the quantum computer `qc`, 
		without the ending 'qvm' if it exists.
		"""
		name = qc.name
		if name[-4:] == "-qvm":
			name = name[0:-4]
			return name
		else:
			return name

	def create_from_other(self, cal):
		self.T1 = cal.T1.copy()
		self.T2 = cal.T2.copy()
		self.fidelity_1q = cal.fidelity_1q.copy()
		self.readout_fidelity = cal.readout_fidelity.copy()
		self.fidelity_CPHASE = cal.fidelity_CPHASE.copy()
		self.fidelity_CZ = cal.fidelity_CZ.copy()
		self.fidelity_XY = cal.fidelity_XY.copy()

	def change_noise_intensity(self, intensity: float):
		self.T1 = self.change_times_by_ratio(self.T1, intensity)
		self.T2 = self.change_times_by_ratio(self.T2, intensity)
		self.fidelity_1q = self.change_fidelity_by_noise_intensity(self.fidelity_1q, intensity)
		self.readout_fidelity = self.change_fidelity_by_noise_intensity(self.readout_fidelity, intensity)
		self.fidelity_CPHASE = self.change_fidelity_by_noise_intensity(self.fidelity_CPHASE, intensity)
		self.fidelity_CZ = self.change_fidelity_by_noise_intensity(self.fidelity_CZ, intensity)
		self.fidelity_XY = self.change_fidelity_by_noise_intensity(self.fidelity_XY, intensity)

	def change_times_by_ratio(self, dict: Dict, ratio: float):
		for key in dict.keys():
			dict[key] = dict[key] / (ratio + 1e-10)
		return dict
	
	def change_fidelity_by_noise_intensity(self, dict:Dict, intensity:float):
		for key in dict.keys():
			dict[key] = max(0.0, min(1.0, 1 - ((1 - dict[key]) * intensity)))
		return dict



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


def create_decoherence_noise_model(
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
	noise_model_2q = create_decoherence_noise_model(
		gates_2q,
		T1=T1,
		T2=T2,
		# for future improvement: use the specific gate time if known
		gate_time=gate_time_2q,
	)
	gates_1q = [i for i in _get_program_gates(prog) if i.name == Noisy_I_1Q_name]
	noise_model_1q = create_decoherence_noise_model(
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
		fidelity: Dict[str, float],
) -> NoiseModel:
	"""
	creates a noise model for depolarizing.

	:param gates: depolarizing gates of a certain type.
	:param fidelity: a mapping betweem qubits (one or a pair) to the fiselity.
	"""

	num_qubits = 1
	all_qubits = []
	for g in gates:
		qubits = [t.index for t in g.qubits]
		if len(qubits) == 1:
			all_qubits.append(str(qubits[0]))
		elif len(qubits) == 2:
			num_qubits = 2
			qubits.sort(key = lambda x: int(x))
			all_qubits.append(str(qubits[0])+'-'+str(qubits[1]))
	all_qubits = set(all_qubits)

	kraus_matrices = {
		q: depolarizing_kraus(num_qubits, fidelity.get(q, 1.0)) for q in all_qubits
	}
	kraus_maps = []
	for g in gates:
		targets = tuple(t.index for t in g.qubits)
		qubits = targets
		if num_qubits == 1:
			qubits = str(qubits[0])
		if num_qubits > 1:
			qubits = sorted(list(targets))
			qubits = str(qubits[0])+'-'+str(qubits[1])
		noisy_I = kraus_matrices[qubits]

		kraus_maps.append(
			KrausModel(
				g.name,
				tuple(g.params),
				targets,
				noisy_I,
				1.0,
			)
		)
	return NoiseModel(kraus_maps, {})


def add_depolarizing_noise(prog: Program, fidelities: Dict[str, Dict[str, float]]):
	"""
	add depolarizing noise to the program.

	:param prog: the program.
	:param fidelities: dictionary of fidelitied by name. each fidelity is a dictionary
	mapping a qubit or a pair of qubits to their fidelity.
	:return: the changed program
	"""

	for name in [Depolarizing_1Q_gate, Depolarizing_CPHASE, Depolarizing_CZ, Depolarizing_XY]:
		gates = [i for i in _get_program_gates(prog) if i.name == name]
		noise_model = create_depolarizing_noise_model(gates, fidelities[name])
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
	noise_model = create_decoherence_noise_model(
		gates,
		T1=T1,
		T2=T2,
		gate_time=gate_info["duration"],
	)
	for k in noise_model.gates:
		prog.define_noisy_gate(k.gate, k.targets, k.kraus_ops)


def _def_noise_gates(name: str, dim: int, new_p: Program):
	"""
	defines a gate wit name `name` for `new_p`, and returns the gate.
	the gate is an identity matrix, in dimention `dim`.

	:param name: gate name.
	:param dim: matrix dimantion.
	:param new_p: the program to add the definitino to.
	:return: the new gate.
	"""
	dg = DefGate(name, np.eye(dim))
	new_p += dg
	return dg.get_constructor()

def define_noisy_gates(
	new_p: Program,
	prog: Program,
	depolarizing: bool,
	decoherence_after_1q_gate: bool,
	decoherence_after_2q_gate: bool,
	# TODO decoherence_only_on_targets: bool,
) -> Dict:
	"""
	defines noisy gates for the new program `new_p`,
	and returns a Dictionary with the new noise gates.
	the function defines noise gates only for types of noises that are given as parameters,
	and only for gates that appear in the program `prog`.

	:param new_p: new program, to add definitions on.
	:param prog: old program, to find which noise gates we need.
	:param depolarizing: add depolarizing noise, default is True.
	:param decoherence_after_1q_gate: add decoherence noise to all qubits after every one-qubit gate.
	:param decoherence_after_2q_gate: add decoherence noise to all qubits after every two-qubit gate.
	:param readout_noise: add readour noise.

	:return: `noise_gates`, a Dictionary with the new noise gates.
	"""

	# check which noise types are needed the program:
	depol_1q, depol_cphase, depol_cz, depol_xy, no_depol = False, False, False, False, False
	dec_1q, dec_2q = False, False
	noise_gates = {}
	for i in prog:
		if isinstance(i, Gate):
			if len(i.qubits) == 1:
				if depolarizing: depol_1q = True
				if decoherence_after_1q_gate: dec_1q = True
			elif len(i.qubits) == 2:
				if decoherence_after_2q_gate: dec_2q = True
				if depolarizing:
					if i.name == "CPHASE": depol_cphase = True
					elif i.name == "CZ": depol_cz = True
					elif i.name == "XY": depol_xy = True
					else: no_depol = True
	
	# add relavent definitions and noise gates:
	if depol_1q: noise_gates[Depolarizing_1Q_gate] = _def_noise_gates(Depolarizing_1Q_gate, 2, new_p)
	if depol_cphase: noise_gates[Depolarizing_CPHASE] = _def_noise_gates(Depolarizing_CPHASE, 4, new_p)
	if depol_cz: noise_gates[Depolarizing_CZ] = _def_noise_gates(Depolarizing_CZ, 4, new_p)
	if depol_xy: noise_gates[Depolarizing_XY] = _def_noise_gates(Depolarizing_XY, 4, new_p)
	if dec_2q: noise_gates[Noisy_I_2Q_name] = _def_noise_gates(Noisy_I_2Q_name, 2, new_p)
	if dec_1q: noise_gates[Noisy_I_1Q_name] = _def_noise_gates(Noisy_I_1Q_name, 2, new_p)
	if no_depol: noise_gates[No_Depolarizing] = _def_noise_gates(No_Depolarizing, 4, new_p)
	return noise_gates


def add_noise_to_program(
		qc: QuantumComputer,
		p: Program,
		convert_to_native: bool = True,
		calibrations: Optional[Calibrations] = None,
		depolarizing: bool = True,
		decoherence_after_1q_gate: bool = False,
		decoherence_after_2q_gate: bool = True,
		# TODO decoherence_only_on_targets: bool = False,
		readout_noise: bool = True,
		noise_intensity: float = 1.0
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
    :param depolarizing: add depolarizing noise, default is True.
	:param decoherence_after_1q_gate: add decoherence noise to all qubits after every one-qubit gate.
	default is False.
	:param decoherence_after_2q_gate: add decoherence noise to all qubits after every two-qubit gate.
	default is True.
	:param readout_noise: add readour noise. default is True.
    :return: A new program with noisy operators.
    """

	if convert_to_native:
		p = qc.compiler.quil_to_native_quil(p)
	qubits = p.get_qubits()

	new_p = Program()

	noise_gates = define_noisy_gates(new_p=new_p, prog=p, 
		depolarizing=depolarizing,
		decoherence_after_2q_gate=decoherence_after_2q_gate, 
		decoherence_after_1q_gate=decoherence_after_1q_gate)
	
	for i in p:
		new_p += i
		if isinstance(i, Gate):
			targets = tuple(t.index for t in i.qubits)
			# for 2-qubit gates, add decoherence noise for all qubits in qc
			if len(targets) == 2:
				if depolarizing:
					# new_p += noise_gates["Depolarizing_" + i.name](targets[0], targets[1]) # change string?
					name = Depolarizing_CPHASE if i.name == "CPHASE" else Depolarizing_CZ if i.name == "CZ" else Depolarizing_XY if i.name == "XY" else No_Depolarizing
					new_p += noise_gates[name](targets[0], targets[1])
				if decoherence_after_2q_gate:
					for q in qubits:
						if q not in targets or not depolarizing:
							new_p += noise_gates[Noisy_I_2Q_name](q)
			elif len(targets) == 1:
				if depolarizing:
					for q in targets:
						new_p += noise_gates[Depolarizing_1Q_gate](targets[0])
				if decoherence_after_1q_gate:
					for q in qubits:
						if q not in targets or not depolarizing:
							new_p += noise_gates[Noisy_I_1Q_name](q)

	# deal with delay:
	new_p, delay_gates = replace_delay_with_noisy_I(new_p)

	if calibrations is None:
		calibrations = Calibrations(qc=qc)
	if noise_intensity != 1.0:
		new_calibrations = Calibrations()
		new_calibrations.create_from_other(calibrations)
		new_calibrations.change_noise_intensity(noise_intensity)
		calibrations = new_calibrations

	# add kraus maps declarations:
	if depolarizing:
		new_p = add_depolarizing_noise(prog=new_p, fidelities={
			Depolarizing_1Q_gate: calibrations.fidelity_1q, 
			Depolarizing_CPHASE: calibrations.fidelity_CPHASE, 
			Depolarizing_CZ: calibrations.fidelity_CZ,
			Depolarizing_XY: calibrations.fidelity_XY})

	if decoherence_after_1q_gate or decoherence_after_2q_gate:
		new_p = add_decoherence_noise(prog=new_p, T1=calibrations.T1, T2=calibrations.T2)
	
	if readout_noise:
		new_p = add_readout_noise(prog=new_p, ro_fidelity=calibrations.readout_fidelity)

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

