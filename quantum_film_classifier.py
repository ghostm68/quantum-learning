import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's numpy for differentiable parameters
import numpy as np  # Regular numpy for other operations
import pandas as pd

print("⚛️ Starting Quantum Film Classifier...")

# Load prepared data
try:
    X = np.load('film_features.npy')
    y = np.load('film_labels.npy')
    print(f"📁 Loaded features: {X.shape}, labels: {y.shape}")
except:
    print("❌ Could not load data files. Run prepare_film_data.py first.")
    exit()

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Quantum circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_classifier(params, x):
    """Quantum circuit for film title classification"""
    # Encode features
    for i in range(min(n_qubits, len(x))):
        qml.RY(x[i] * np.pi, wires=i)
    
    # Variational layers
    for layer in range(2):
        for i in range(n_qubits):
            qml.RY(params[layer, i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
    
    return qml.expval(qml.PauliZ(0))

# Simple training
print("🎯 Training on 100 film samples...")
params = pnp.random.uniform(0, 2*np.pi, size=(2, n_qubits), requires_grad=True)  # FIXED LINE

# Use a small subset for quick testing
train_size = min(100, len(X))
X_train, y_train = X[:train_size], y[:train_size]

from pennylane.optimize import GradientDescentOptimizer
opt = GradientDescentOptimizer(stepsize=0.1)

for epoch in range(30):
    total_cost = 0
    for i in range(train_size):
        def cost(p):
            pred = quantum_classifier(p, X_train[i])
            return ((pred + 1) / 2 - y_train[i]) ** 2
        
        params = opt.step(cost, params)
        total_cost += cost(params)
    
    if epoch % 10 == 0:
        print(f"   Epoch {epoch}: Avg cost = {total_cost/train_size:.4f}")

print("\n✅ Training complete!")
print("\n📊 Testing on 5 films:")
df = pd.read_csv('enriched_films.csv')
for i in range(5):
    pred = quantum_classifier(params, X[i])
    pred_label = 1 if pred > 0 else 0
    actual = y[i]
    status = "✓" if pred_label == actual else "✗"
    print(f"   {status} '{df.iloc[i]['TITLE'][:30]}...' → QNN: {pred:.3f} (pred: {pred_label}, actual: {actual})")
