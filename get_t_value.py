import requests
import json

def get_t_value(QPU: int=2, qubit: int=0, T: int=1) -> float:
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
    intqs = [int(q) for q in qs]
    return dict(zip(intqs,t1)), dict(zip(intqs,t2))