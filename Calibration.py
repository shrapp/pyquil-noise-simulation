import requests
import json

class Calibration:
    """
    encapsulate the calibration data for Aspen-M-2 or Aspen-M-3 machine.

    args: QPU (int, optional): Choose between 2 or 3 (Aspen-M-2\Aspen-M-3) Defaults to 2.
    """

    def __init__(self, QPU: int=2) -> None:
        if QPU not in [2,3]:
            raise ValueError("QPU must be 2 or 3")
        else:
            url = "https://forest-server.qcs.rigetti.com/lattices/Aspen-M-"
            response = requests.get(url + str(QPU))
            file = json.loads(response.text)
            self.calibrations = file["lattice"]["specs"]
        
    
    def T1(self, Q: int=0) -> float:
        """
        Get the T1 value for a given qubit.

        Args:
            Q (int, optional): Choose qubit id, Defaults to 0.

        Raises:
            ValueError: Qubit not found

        Returns:
            float: T1 time in microseconds
        """
        if str(Q) not in self.calibrations["1Q"]:
            raise ValueError("Qubit not found")
        return self.calibrations["1Q"][str(Q)]["T1"]

    def T2(self, Q: int) -> float:
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
            gate (string): The gate that we wan't its fidelity.

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
            if "f"+gate in self.calibrations["2Q"][pair]:
                gate = "f"+gate
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
        if str(Q1)+"-"+str(Q2) in QQ:
            return str(Q1)+"-"+str(Q2)
        elif str(Q2)+"-"+str(Q2) in QQ:
            return str(Q2)+"-"+str(Q1)
        else:
            return None

