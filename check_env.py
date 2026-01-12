import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Check if we're in a virtual environment
import os
print("VIRTUAL_ENV:", os.environ.get('VIRTUAL_ENV', 'Not in venv'))

# Try to import PennyLane
try:
    import pennylane as qml
    print("✅ PennyLane:", qml.__version__)
except ImportError as e:
    print("❌ PennyLane import error:", e)
