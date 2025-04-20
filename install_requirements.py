import subprocess
import sys
import os

def install_from_requirements(requirements_file="requirements.txt"):
    if not os.path.exists(requirements_file):
        print(f"{requirements_file} not found.")
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All requirements installed successfully.")
    except subprocess.CalledProcessError:
        print("An error occurred while installing requirements.")

if __name__ == "__main__":
    install_from_requirements()
