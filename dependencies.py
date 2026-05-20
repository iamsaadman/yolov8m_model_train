import subprocess, sys

# Remove incompatible torch
subprocess.run([
    sys.executable, "-m", "pip", "uninstall",
    "-y", "torch", "torchvision", "torchaudio"
], check=True)

# Install compatible PyTorch for P100
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu121"
], check=True)

# Install Ultralytics
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "ultralytics"
], check=True)

print("✅ Correct P100-compatible setup installed")
print("⚠️ Restart Runtime now")
