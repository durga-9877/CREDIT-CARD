import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install base packages first
base_packages = [
    "setuptools>=68.2.2",
    "wheel>=0.41.2",
    "numpy>=1.26.0",  # Latest version compatible with Python 3.12
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.1",
    "imbalanced-learn>=0.10.1",
    "flask>=2.0.1",
    "python-dotenv>=1.0.0",
    "pytest>=7.4.2",
    "gunicorn>=21.2.0"
]

for package in base_packages:
    try:
        install_package(package)
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        sys.exit(1)

print("All dependencies installed successfully!") 