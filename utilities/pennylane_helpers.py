"""
PennyLane-specific helper functions
"""
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def create_parameterized_circuit(n_qubits, n_layers):
    """
    Create a parameterized quantum circuit (ansatz)
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        
    Returns:
        Function that creates the circuit
    """
    def circuit(params, inputs=None):
        # Encode inputs if provided
        if inputs is not None:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
        
        # Apply layers
        for layer in range(n_layers):
            # Rotations
            for qubit in range(n_qubits):
                qml.RY(params[layer, qubit, 0], wires=qubit)
                qml.RZ(params[layer, qubit, 1], wires=qubit)
            
            # Entanglement
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
    
    return circuit

def run_on_device(circuit_fn, device_name="default.qubit", **device_kwargs):
    """
    Run a circuit on a specific device
    
    Args:
        circuit_fn: Function that creates the circuit
        device_name: Name of the device
        **device_kwargs: Additional device arguments
        
    Returns:
        QNode ready for execution
    """
    # Get number of wires from circuit
    import inspect
    source = inspect.getsource(circuit_fn)
    wires = source.count('wires=')
    
    dev = qml.device(device_name, wires=wires, **device_kwargs)
    return qml.QNode(circuit_fn, dev)

def compare_simulators(circuit_fn, params, inputs=None):
    """
    Compare circuit execution on different simulators
    """
    simulators = ["default.qubit", "lightning.qubit"]
    
    results = {}
    for sim in simulators:
        try:
            qnode = run_on_device(circuit_fn, device_name=sim)
            results[sim] = qnode(params, inputs)
        except Exception as e:
            print(f"Error with {sim}: {e}")
    
    return results
