import subprocess
import sys

print("Installing Jupyter extension requirements...")

# Install Jupyter and ipykernel
subprocess.run([sys.executable, "-m", "pip", "install", 
               "jupyter", "ipykernel", "jupyterlab"], 
               capture_output=True, text=True)

print("✅ Jupyter components installed")

# Create a kernel
subprocess.run([sys.executable, "-m", "ipykernel", "install", 
               "--user", "--name", "python3", "--display-name", "Python 3"],
               capture_output=True, text=True)

print("✅ Python kernel installed")
print("\nNow restart Codespace or reload window (Ctrl+Shift+P → 'Developer: Reload Window')")
