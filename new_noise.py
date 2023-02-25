##############################################################################
# Written by Shahar Rapp and Ze'ev Binnes, added on Rigetti's pyquil.
#
#    A new noise model based on pyquil.noise,
#    which works by adding "noisy I" gates after long operations.
##############################################################################

import json
from typing import Dict, List, Sequence, Optional, Any

import numpy as np
import requests
from pyquil.api import QuantumComputer
from pyquil.noise import _get_program_gates, INFINITY, pauli_kraus_map, damping_kraus_map, \
	dephasing_kraus_map, combine_kraus_maps
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Gate, DefGate, Pragma, DelayQubits

Noisy_I_1Q_name = "Damping_and_dephasing_for_1Q_gate"
Noisy_I_2Q_name = "Damping_and_dephasing_for_2Q_gate"
Depolarizing_1Q_gate = "Depolarizing_1Q_gate"
Depolarizing_CPHASE = "Depolarizing_CPHASE"
Depolarizing_CZ = "Depolarizing_CZ"
Depolarizing_XY = "Depolarizing_XY"
No_Depolarizing = "No_Depolarizing"


def get_qc_name(qc: QuantumComputer):
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


def change_times_by_ratio(times: Dict, ratio: float):
	for key in times.keys():
		times[key] = times[key] / (ratio + 1e-10)
	return times


def change_fidelity_by_noise_intensity(fidelity: Dict, intensity: float):
	for key in fidelity.keys():
		fidelity[key] = max(0.0, min(1.0, 1 - ((1 - fidelity[key]) * intensity)))
	return fidelity


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

	def __init__(self, qc: Optional[QuantumComputer] = None):
		self.fidelity_XY = None
		self.fidelity_CZ = None
		self.fidelity_CPHASE = None
		self.readout_fidelity = None
		self.fidelity_1q = None
		self.T2 = None
		self.T1 = None
		if qc is None:
			return  # user can set his own values
		else:
			qc_name = get_qc_name(qc)
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

	def create_from_other(self, cal):
		self.T1 = cal.T1.copy()
		self.T2 = cal.T2.copy()
		self.fidelity_1q = cal.fidelity_1q.copy()
		self.readout_fidelity = cal.readout_fidelity.copy()
		self.fidelity_CPHASE = cal.fidelity_CPHASE.copy()
		self.fidelity_CZ = cal.fidelity_CZ.copy()
		self.fidelity_XY = cal.fidelity_XY.copy()

	def change_noise_intensity(self, intensity: float):
		self.T1 = change_times_by_ratio(self.T1, intensity)
		self.T2 = change_times_by_ratio(self.T2, intensity)
		self.fidelity_1q = change_fidelity_by_noise_intensity(self.fidelity_1q, intensity)
		self.readout_fidelity = change_fidelity_by_noise_intensity(self.readout_fidelity, intensity)
		self.fidelity_CPHASE = change_fidelity_by_noise_intensity(self.fidelity_CPHASE, intensity)
		self.fidelity_CZ = change_fidelity_by_noise_intensity(self.fidelity_CZ, intensity)
		self.fidelity_XY = change_fidelity_by_noise_intensity(self.fidelity_XY, intensity)


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
) -> List[Dict[str, Any]]:
	"""
    Create A List with the appropriate Kraus operators defined.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time dictionary indexed by qubit id.
    :param T2: The T2 dephasing time dictionary indexed by qubit id.
    :param gate_time: The duration of a gate
    :return: A List with the appropriate Kraus operators defined.
    """
	all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))

	matrices = {
		q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time) for q in all_qubits
	}
	kraus_maps = []
	for g in gates:
		targets = tuple(t.index for t in g.qubits)
		kraus_ops = matrices[targets[0]]

		kraus_maps.append({"gate": g.name, "params": tuple(g.params), "targets": targets, "kraus_ops": kraus_ops})
	return kraus_maps


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
    :param gate_time_1q: The duration of the one-qubit gates. By default, this is 32 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ. By default, this is 176 ns.
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
	for k in noise_model_2q:
		prog.define_noisy_gate(k["gate"], k["targets"], k["kraus_ops"])
	for k in noise_model_1q:
		prog.define_noisy_gate(k["gate"], k["targets"], k["kraus_ops"])
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
	# add readout noise pragmas
	for q, ap in assignment_probs.items():
		prog.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
	return prog


def create_depolarizing_noise_model(
		gates: Sequence[Gate],
		fidelity: Dict[str, float],
) -> List[Dict[str, Any]]:
	"""
	creates a noise model for depolarizing.

	:param gates: depolarizing gates of a certain type.
	:param fidelity: a mapping between qubits (one or a pair) to the fidelity.
	"""

	num_qubits = 1
	all_qubits = []
	for g in gates:
		qubits = [t.index for t in g.qubits]
		if len(qubits) == 1:
			all_qubits.append(str(qubits[0]))
		elif len(qubits) == 2:
			num_qubits = 2
			qubits.sort(key=lambda x: int(x))
			all_qubits.append(str(qubits[0]) + '-' + str(qubits[1]))
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
			qubits = str(qubits[0]) + '-' + str(qubits[1])
		kraus_ops = kraus_matrices[qubits]

		kraus_maps.append({"gate": g.name, "params": tuple(g.params), "targets": targets, "kraus_ops": kraus_ops})
	return kraus_maps


def add_depolarizing_noise(prog: Program, fidelities: Dict[str, Dict[str, float]]):
	"""
	add depolarizing noise to the program.

	:param prog: the program.
	:param fidelities: dictionary of fidelities by name. each fidelity is a dictionary
	mapping a qubit or a pair of qubits to their fidelity.
	:return: the changed program
	"""

	for name in [Depolarizing_1Q_gate, Depolarizing_CPHASE, Depolarizing_CZ, Depolarizing_XY]:
		gates = [i for i in _get_program_gates(prog) if i.name == name]
		noise_model = create_depolarizing_noise_model(gates, fidelities[name])
		for k in noise_model:
			prog.define_noisy_gate(k["gate"], k["targets"], k["kraus_ops"])
	return prog


def add_delay_maps(prog: Program, delay_gates: Dict[str, float], T1: Dict[int, float], T2: Dict[int, float]) -> Program:
	"""
    Add kraus maps for a `DELAY` instruction,
    that was converted already into `noisy-I` gate.

    :param prog: the program to add the maps to.
    :param delay_gates: a Dictionary with the gates name and duration.
    :param T1: Dictionary with T1 times.
    :param T2: Dictionary with T2 times.
    """
	if len(delay_gates.items()) > 0:
		for name, duration in delay_gates.items():
			gates = [i for i in _get_program_gates(prog) if i.name == name]
			noise_model = create_decoherence_noise_model(
				gates,
				T1=T1,
				T2=T2,
				gate_time=duration,
			)
			for k in noise_model:
				prog.define_noisy_gate(k["gate"], k["targets"], k["kraus_ops"])
	return prog


def def_gate(name: str, dim: int, new_p: Program):
	"""
	defines a gate wit name `name` for `new_p`, and returns the gate.
	the gate is an identity matrix, in dimension `dim`.

	:param name: gate name.
	:param dim: matrix dimension.
	:param new_p: the program to add the definition to.
	:return: the new gate.
	"""
	dg = DefGate(name, np.eye(dim))
	# TODO: Do we have to add the definitions?
	new_p += dg
	return dg.get_constructor()


def define_noisy_gates(
		new_p: Program,
		prog: Program,
		depolarizing: bool,
		decoherence_after_1q_gate: bool,
		decoherence_after_2q_gate: bool
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

	:return: `noise_gates`, a Dictionary with the new noise gates.
	"""

	# check which noise types are needed the program:
	depol_1q = depol_cphase = depol_cz = depol_xy = no_depol = dec_1q = dec_2q = False
	noise_gates = {}
	for i in prog:
		if (decoherence_after_1q_gate or decoherence_after_2q_gate) and \
				((isinstance(i, Pragma) and i.command == "DELAY") or isinstance(i, DelayQubits)):
			duration = i.duration if isinstance(i, DelayQubits) else i.freeform_string
			name = "Noisy-DELAY-" + duration
			if name not in noise_gates.keys():
				noise_gates[name] = def_gate(name, 2, new_p)
		if isinstance(i, Gate):
			if len(i.qubits) == 1:
				if depolarizing:
					depol_1q = True
				if decoherence_after_1q_gate:
					dec_1q = True
			elif len(i.qubits) == 2:
				if decoherence_after_2q_gate:
					dec_2q = True
				if depolarizing:
					if i.name == "CPHASE":
						depol_cphase = True
					elif i.name == "CZ":
						depol_cz = True
					elif i.name == "XY":
						depol_xy = True
					else:
						no_depol = True

	# add relevant definitions and noise gates:
	if depol_1q:
		noise_gates[Depolarizing_1Q_gate] = def_gate(Depolarizing_1Q_gate, 2, new_p)
	if depol_cphase:
		noise_gates[Depolarizing_CPHASE] = def_gate(Depolarizing_CPHASE, 4, new_p)
	if depol_cz:
		noise_gates[Depolarizing_CZ] = def_gate(Depolarizing_CZ, 4, new_p)
	if depol_xy:
		noise_gates[Depolarizing_XY] = def_gate(Depolarizing_XY, 4, new_p)
	if dec_2q:
		noise_gates[Noisy_I_2Q_name] = def_gate(Noisy_I_2Q_name, 2, new_p)
	if dec_1q:
		noise_gates[Noisy_I_1Q_name] = def_gate(Noisy_I_1Q_name, 2, new_p)
	if no_depol:
		noise_gates[No_Depolarizing] = def_gate(No_Depolarizing, 4, new_p)
	return noise_gates


def add_noise_to_program(
		qc: QuantumComputer,
		p: Program,
		convert_to_native: bool = True,
		calibrations: Optional[Calibrations] = None,
		depolarizing: bool = True,
		decoherence_after_1q_gate: bool = False,
		decoherence_after_2q_gate: bool = True,
		# TODO: decoherence_only_on_targets: bool = False,
		readout_noise: bool = True,
		noise_intensity: float = 1.0
) -> Program:
	"""
    Add generic damping and dephasing noise to a program.
    Noise is added to all qubits, after a 2-qubit gate operation.
    This function will define new "I" gates and add Kraus noise to these gates.
    :param noise_intensity: one parameter to control the noise intensity.
    :param qc: A Quantum computer object
    :param p: A pyquil program
    :param convert_to_native: put `False` if the program is already in native pyquil or is not needed -
    Note that it removes any delays.
    :param calibrations: optional, can get the calibrations in advance,
        instead of producing them from the URL.
    :param depolarizing: add depolarizing noise, default is True.
	:param decoherence_after_1q_gate: add decoherence noise to all qubits after every one-qubit gate.
	default is False.
	:param decoherence_after_2q_gate: add decoherence noise to all qubits after every two-qubit gate.
	default is True.
	:param readout_noise: add readout noise. default is True.
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

	delay_gates = {}

	for i in p:
		if (decoherence_after_1q_gate or decoherence_after_2q_gate) and (
				(isinstance(i, Pragma) and i.command == "DELAY") or isinstance(i, DelayQubits)):
			duration = i.duration if isinstance(i, DelayQubits) else i.freeform_string
			name = "Noisy-DELAY-" + duration
			qubits = i.qubits if isinstance(i, DelayQubits) else i.args
			for q in qubits:
				new_p += noise_gates[name](Qubit(q))
			if name not in delay_gates.keys():
				delay_gates[name] = float(duration)

		else:
			new_p += i
			if isinstance(i, Gate):
				targets = tuple(t.index for t in i.qubits)

				if len(targets) == 2:
					if depolarizing:
						# TODO: import gates options from calibrations
						name = ("Depolarizing_" + i.name) if i.name in ['CPHASE', "XY", "CZ"] else No_Depolarizing
						new_p += noise_gates[name](targets[0], targets[1])
					if decoherence_after_2q_gate:
						for q in qubits:
							if q not in targets or not depolarizing:
								new_p += noise_gates[Noisy_I_2Q_name](q)

				elif len(targets) == 1:
					if depolarizing:
						new_p += noise_gates[Depolarizing_1Q_gate](targets[0])
					if decoherence_after_1q_gate:
						for q in qubits:
							if q not in targets or not depolarizing:
								new_p += noise_gates[Noisy_I_1Q_name](q)

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

	new_p = add_delay_maps(new_p, delay_gates, calibrations.T1, calibrations.T2)

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
