#!/usr/bin/env python3
"""
Quantum Computing Test with PennyLane
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 50)
    print("QUANTUM COMPUTING TEST")
    print("=" * 50)
    
    print(f"PennyLane version: {qml.__version__}")
    
    # Create quantum device
    dev = qml.device("default.qubit", wires=2)
    
    # Define circuit
    @qml.qnode(dev)
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    # Execute
    params = np.array([0.5, 0.3])
    result = circuit(params)
    print(f"\n📊 Expectation value: {result}")
    
    # Gradient
    gradient = qml.grad(circuit)(params)
    print(f"🎯 Gradient: {gradient}")
    
    # Visualize (save instead of show)
    print("\n📈 Saving circuit visualization...")
    fig, ax = qml.draw_mpl(circuit)(params)
    plt.savefig("circuit.png")
    print("✅ Saved as circuit.png")
    
    print("\n🎉 Quantum test successful!")

if __name__ == "__main__":
    main()
