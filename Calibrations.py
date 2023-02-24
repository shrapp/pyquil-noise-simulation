import json
from typing import Optional

import requests
from pyquil.api import QuantumComputer


class Calibrations:
	"""
    encapsulate the calibration data for Aspen-M-2 or Aspen-M-3 machine.

    args: qc (QuantumComputer, optional): a Quantum Computer (Aspen-M-2 or Aspen-M-3). 
    Defaults to None, where the user can define his own calibration data.
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


	def t1(self, Q: int = 0) -> float:
		"""
        Get the T1 value for a given qubit.

        Args:
            Q (int, optional): Choose qubit id, Defaults to 0.`

        Raises:
            ValueError: Qubit not found

        Returns:
            float: T1 time in microseconds
        """
		if str(Q) not in self.calibrations["1Q"]:
			raise ValueError("Qubit not found")
		return self.calibrations["1Q"][str(Q)]["T1"]

	def t2(self, Q: int) -> float:
		"""
        Get the T2 value for a given qubit.

        Args:
            Q (int, optional): Choose qubit id, Defaults to 0.

        Raises:
            ValueError: Qubit not found

        Returns:
            float: T2 time in microseconds
        """
		if str(Q) not in self.calibrations["1Q"]:
			raise ValueError("Qubit not found")
		return self.calibrations["1Q"][str(Q)]["T2"]

	def one_qubit_fidelity(self, Q=0, fidelity="f1QRB") -> float:
		"""
        Get the fidelity for a given qubit.

        Args:
            Q (int, optional): Choose qubit id, Defaults to 0.
            fidelity (string, optional): type of fidelity, Defaults to "f1QRB".

        Raises:
            ValueError: Qubit not found
            ValueError: No such fidelity for this qubit

        Returns:
            float: fidelity time in microseconds
        """
		if str(Q) not in self.calibrations["1Q"]:
			raise ValueError("Qubit not found")
		if fidelity not in self.calibrations["1Q"][str(Q)]:
			raise ValueError("No such fidelity for this qubit")
		return self.calibrations["1Q"][str(Q)][fidelity]

	def two_qubit_fidelity(self, Q1: int, Q2: int, gate) -> float:
		"""
        Get the fidelity for a given pair of qubits for a given gate.

        Args:
            Q1 (int): Choose qubit id.
            Q2 (int): Choose qubit id.
            gate (string): The gate that we want its fidelity.

        Raises:
            ValueError: Qubit not found
            ValueError: No such gate for this pair

        Returns:
            float: fidelity time in microseconds
        """
		pair = self.find_pair_qubits(Q1, Q2)
		if pair is None:
			raise ValueError("Qubits not found")
		if gate not in self.calibrations["2Q"][pair]:
			if "f" + gate in self.calibrations["2Q"][pair]:
				gate = "f" + gate
			else:
				raise ValueError("No such gate for this pair")
		return self.calibrations["2Q"][pair][gate]

	def fCPHASE(self, Q1: int, Q2: int) -> float:
		pair = self.find_pair_qubits(Q1, Q2)
		if pair is None:
			raise ValueError("Qubits not found")
		if "fCPHASE" not in self.calibrations["2Q"][pair]:
			raise ValueError("No such gate for this pair")
		return self.calibrations["2Q"][pair]["fCPHASE"]

	def fCZ(self, Q1: int, Q2: int) -> float:
		pair = self.find_pair_qubits(Q1, Q2)
		if pair is None:
			raise ValueError("Qubits not found")
		if "fCZ" not in self.calibrations["2Q"][pair]:
			raise ValueError("No such gate for this pair")
		return self.calibrations["2Q"][pair]["fCZ"]

	def fXY(self, Q1: int, Q2: int) -> float:
		pair = self.find_pair_qubits(Q1, Q2)
		if pair is None:
			raise ValueError("Qubits not found")
		if "fXY" not in self.calibrations["2Q"][pair]:
			raise ValueError("No such gate for this pair")
		return self.calibrations["2Q"][pair]["fXY"]

	def find_pair_qubits(self, Q1, Q2):
		QQ = self.calibrations["2Q"]
		if str(Q1) + "-" + str(Q2) in QQ:
			return str(Q1) + "-" + str(Q2)
		elif str(Q2) + "-" + str(Q2) in QQ:
			return str(Q2) + "-" + str(Q1)
		else:
			return None


def get_t_value(QPU: int = 2, qubit: int = 0, T: int = 1) -> float:
	"""
    Get the T1 or T2 value for a given qubit on a given QPU.

    Args:
        QPU (int, optional): Choose between 2 or 3 (Aspen-M-2, Aspen-M-3) Defaults to 2.
        qubit (int, optional): Choose qubit id, Defaults to 0.
        T (int, optional): Choose between 1 or 2 (T1,T2). Defaults to 1.

    Raises:
        ValueError: T must be 1 or 2
        ValueError: QPU must be 2 or 3
        ValueError: Qubit not found

    Returns:
        float: T1/2 time in microseconds
    """

	url = "https://forest-server.qcs.rigetti.com/lattices/Aspen-M-"

	if T not in [1, 2]:
		raise ValueError("T must be 1 or 2")

	if QPU not in [2, 3]:
		raise ValueError("QPU must be 2 or 3")

	response = requests.get(url + str(QPU))

	file = json.loads(response.text)

	qs = file["lattice"]["specs"]["1Q"]

	if str(qubit) not in qs:
		raise ValueError("Qubit not found")

	return qs[str(qubit)]["T" + str(T)]


def get_T_values(qc_name="Aspen-M-2"):
	"""
    Get the T1 or T2 value for a given qubit on a given QPU.

    Args:
        qc_name: name of the qc
        
    Returns:
        float: T1/2 time in microseconds
    """
	url = "https://forest-server.qcs.rigetti.com/lattices/"
	response = requests.get(url + qc_name)
	file = json.loads(response.text)
	calibrations = file["lattice"]["specs"]
	qs = calibrations['1Q'].keys()
	t1 = [calibrations['1Q'][q]['T1'] for q in qs]
	t2 = [calibrations['1Q'][q]['T2'] for q in qs]
	qubits_indexes = [int(q) for q in qs]
	return dict(zip(qubits_indexes, t1)), dict(zip(qubits_indexes, t2))


def get_readout_fidelity(qc_name="Aspen-M-2"):
	url = "https://forest-server.qcs.rigetti.com/lattices/"
	response = requests.get(url + qc_name)
	file = json.loads(response.text)
	calibrations = file["lattice"]["specs"]
	qs = calibrations['1Q'].keys()
	readout = [calibrations['1Q'][q]["fRO"] for q in qs]
	qubits_indexes = [int(q) for q in qs]
	return dict(zip(qubits_indexes, readout))


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
