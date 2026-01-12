import pennylane as qml
import numpy as np

# Load prepared data
X = np.load('film_features.npy')
y = np.load('film_labels.npy')

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Quantum circuit for classification
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_classifier(params, x):
    """Quantum circuit that processes film title features"""
    # Encode features (using first 4 features for 4 qubits)
    for i in range(n_qubits):
        qml.RY(x[i] * np.pi, wires=i)
    
    # Variational layers (trainable)
    for layer in range(2):
        for i in range(n_qubits):
            qml.RY(params[layer, i], wires=i)
        # Entanglement
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
    
    return qml.expval(qml.PauliZ(0))

# Training setup
from pennylane.optimize import AdamOptimizer

def train_quantum_classifier(X_train, y_train, epochs=100):
    params = np.random.uniform(0, 2*np.pi, size=(2, n_qubits), requires_grad=True)
    opt = AdamOptimizer(stepsize=0.1)
    
    for epoch in range(epochs):
        batch_idx = np.random.choice(len(X_train), size=32)
        X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
        
        def cost(p):
            predictions = np.array([quantum_classifier(p, x) for x in X_batch])
            predictions = (predictions + 1) / 2  # Convert to [0, 1]
            return np.mean((predictions - y_batch) ** 2)
        
        params = opt.step(cost, params)
        
        if epoch % 20 == 0:
            current_cost = cost(params)
            print(f"Epoch {epoch}: Cost = {current_cost:.4f}")
    
    return params

# Train on a subset
print("Training quantum classifier on film title patterns...")
trained_params = train_quantum_classifier(X[:500], y[:500], epochs=50)

# Test predictions
test_idx = 505
test_features = X[test_idx]
prediction = quantum_classifier(trained_params, test_features)
actual = y[test_idx]
print(f"\nTest prediction: {prediction:.3f} (actual: {actual})")
print(f"Film title: {pd.read_csv('enriched_films.csv').iloc[test_idx]['TITLE']}")
