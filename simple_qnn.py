import pennylane as qml
from pennylane import numpy as np  # PennyLane's NumPy for automatic gradients
import matplotlib.pyplot as plt

# ===== 1. Define the Quantum Circuit (The QNN's "Layer") =====
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def circuit(params, x=None):
    """A simple parameterized quantum circuit.
       params: The "weights" to train.
       x: A single input data point (for simplicity).
    """
    # Encode the input data onto the qubits (this is "input encoding")
    for i in range(n_qubits):
        qml.RY(x, wires=i) if x is not None else qml.Hadamard(wires=i)
    
    # Apply tunable parameters (this is the "variational layer")
    # We'll use a simple rotation on each qubit
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    
    # Entangle the qubits to create complexity
    qml.CNOT(wires=[0, 1])
    
    # Return the expectation value of a measurement as the output
    return qml.expval(qml.PauliZ(0))

# Make the circuit executable
qnode = qml.QNode(circuit, dev)

# ===== 2. Create a Toy Learning Task =====
# Let's teach the QNN to output +1 for input 0.5 and -1 for input -0.5
X_train = np.array([-0.5, 0.5])
Y_train = np.array([-1.0, 1.0])

# ===== 3. The Cost Function (What we want to minimize) =====
def cost(params, X, Y):
    """Mean squared error between QNN predictions and true labels."""
    predictions = np.array([qnode(params, x=x) for x in X])
    return np.mean((predictions - Y) ** 2)

# ===== 4. Training the QNN =====
# Initialize random parameters (the "weights" we will train)
params = np.random.uniform(0, 2*np.pi, size=n_qubits, requires_grad=True)

# Choose an optimizer (the classical algorithm that updates params)
opt = qml.GradientDescentOptimizer(stepsize=0.3)

cost_history = []
print("Starting training...")
for epoch in range(50):
    # One step of gradient descent
    params, current_cost = opt.step_and_cost(cost, params, X_train, Y_train)
    cost_history.append(current_cost)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Cost = {current_cost:.7f}")

print(f"\nFinal parameters: {params}")

# ===== 5. Check the Learned Function =====
# See what the QNN outputs for a range of inputs
X_test = np.linspace(-1, 1, 50)
predictions = np.array([qnode(params, x=x) for x in X_test])

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(X_train, Y_train, 'ro', label='Training data', markersize=10)
plt.plot(X_test, predictions, 'b-', label='QNN prediction', linewidth=2)
plt.xlabel('Input x')
plt.ylabel('QNN Output')
plt.title('Simple Quantum Neural Network - Learned Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('my_first_qnn.png')  # Saves the plot so you can view it
print("\n✅ Plot saved as 'my_first_qnn.png'. Check your file explorer!")
