# Superconducting Neural Network for Image Classification

A quantum machine learning project implementing Flexible Representation of Quantum Images (FRQI) encoding for binary image classification using quantum neural networks. This project uses different quantum computing frameworks (Qiskit, PennyLane, and Qadence) for quantum image processing and classification tasks.

## Overview

This project explores the intersection of quantum computing and machine learning by implementing quantum neural networks for image classification. The main task is to classify synthetic images based on the presence or absence of elliptical shapes, demonstrating the potential of quantum algorithms for computer vision applications.

### Key Features

- **FRQI Encoding**: Implementation of Flexible Representation of Quantum Images for efficient quantum image encoding
- **Multi-Framework Support**: Implementations using three major quantum computing frameworks:
  - Qiskit with Qiskit Machine Learning
  - PennyLane
  - Qadence
- **Hybrid Architecture**: Classical-quantum hybrid neural networks combining quantum feature encoding with classical processing
- **Comprehensive Evaluation**: Detailed performance metrics including accuracy, precision, recall, F1-score, and resource usage analysis
- **Synthetic Dataset**: Automated generation of training and test datasets with controllable complexity

## Architecture

### FRQI (Flexible Representation of Quantum Images)

The project implements FRQI encoding which represents classical images in quantum states:

- **Position Qubits**: Encode pixel positions using binary representation
- **Color Qubit**: Encodes grayscale intensity values using rotation angles
- **Multi-Controlled Gates**: Implement precise pixel-wise quantum state preparation

### Quantum Circuit Components

1. **Feature Encoding Circuit**: FRQI-based quantum image encoding
2. **Variational Ansatz**: Parameterized quantum circuits for learning
3. **Measurement**: Observable-based output extraction
4. **Classical Processing**: Post-processing for final classification

## Project Structure

```
src/
├── Qiskit.py                    # Qiskit implementation with full training pipeline
├── Qiskit - FRQI.ipynb         # Jupyter notebook for Qiskit FRQI exploration
├── Pennylane - FRQI.ipynb      # PennyLane implementation and visualization
├── Qadence.py                  # Qadence framework implementation
└── Qadence/                    # Additional Qadence experiments
```

## Getting Started

### Prerequisites

```bash
# Core quantum computing frameworks
pip install qiskit qiskit-machine-learning
pip install pennylane
pip install qadence

# Machine learning and data processing
pip install torch torchvision
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install tqdm pillow

# Additional dependencies
pip install pylatexenc  # For Qiskit circuit visualization
```

### Running the Experiments

#### Qiskit Implementation

```bash
python src/Qiskit.py --data-dir ./dataset --epochs 50 --batch-size 32 --output-dir ./results
```

#### Qadence Implementation

```bash
python src/Qadence.py --image_size 4 --depth 3 --epochs 100 --results_dir ./results
```

#### Jupyter Notebooks

Launch Jupyter and explore the interactive notebooks for detailed analysis and visualization:

```bash
jupyter notebook src/
```

### Command Line Arguments

#### Qiskit.py

- `--data-dir`: Directory containing training data with labels.csv
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Results output directory

#### Qadence.py

- `--image_size`: Image dimensions (default: 4x4)
- `--depth`: Quantum circuit depth (default: 3)
- `--epochs`: Training epochs (default: 10)
- `--results_dir`: Output directory for results
