# Quantum Learning with PennyLane ⚛️

A clean, focused quantum computing repository using **only PennyLane**.

## Why PennyLane?

1. **Hardware-Agnostic**: Same code runs on IBM, Google, IonQ, Rigetti, OQC (via OQTOPUS)
2. **Quantum ML Focus**: Built for variational algorithms and quantum neural networks
3. **Automatic Differentiation**: Differentiate quantum circuits like neural networks
4. **Modern API**: Clean, intuitive interface similar to PyTorch/TensorFlow

## What You Can Do Right Now

### ✅ Definitely Working:
- Local quantum simulations (30+ qubits)
- Quantum gradients and optimization
- Variational Quantum Algorithms (VQE, QAOA)
- Quantum Machine Learning models
- OQTOPUS integration (your existing setup)

### 🔧 Real Hardware Access:
- **OQTOPUS**: European quantum ecosystem (your current setup)
- **PennyLane AI**: Cloud quantum computing
- **IBM Quantum**: Via PennyLane plugin (optional)
- **Multiple backends**: Through PennyLane's unified interface

## Getting Started

### 1. Open in GitHub Codespaces (Recommended)
- Click "Code" → "Codespaces" → "New codespace"
- Wait 60 seconds for PennyLane to load
- Open `notebooks/01_pennylane_basics.ipynb`

### 2. Local Development
```bash
git clone https://github.com/yourusername/quantum-learning
cd quantum-learning
pip install -r requirements.txt
jupyter lab
