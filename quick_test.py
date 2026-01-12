import pennylane as qml
print(f"✅ PennyLane {qml.__version__} installed!")

# Create a simple circuit
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    return qml.state()

print(f"Quantum state: {circuit()}")
