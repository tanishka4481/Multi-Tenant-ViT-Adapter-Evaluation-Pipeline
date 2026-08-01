import os
import sys
import subprocess

# 1. Force reinstall 4.49.0 so all cached files are overwritten
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
    "transformers==4.49.0", "accelerate", "roboflow", "scikit-learn", "matplotlib", "timm", "torchmetrics"
])

print("✅ Installation complete. Restarting runtime to apply module changes...")

# 2. Automatically restart Python runtime cleanly
os._exit(0)
