import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers==4.49.0", "peft", "accelerate", "einops", "timm"])
