#Run as; "python qnn_train4by4_timed.py --data-dir ./dataset_4x4_1024 --epochs 20 --batch-size 32 --output-dir ./results"

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import resource

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RYGate
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# ========== CONFIGURABLE CONSTANTS ==========
IMAGE_SIZE = 4
GRAYSCALE_LEVELS = 8
SEED = 42

# ========== DATA LOADING ==========
def load_image_dataset(labels_file, img_size):
    df = pd.read_csv(labels_file)  #Read the CSV file with the paths of the images and with the labels
    images, labels = [], []  #Initialize lists to store the images and the labels
    for _, row in df.iterrows():  #Process each image
        img = Image.open(row['filename']).convert('L')  #Load the image
        img = img.resize((img_size, img_size))  
        img_array = np.array(img).flatten() / 255.0  #Normalize for [0, 1]
        images.append(img_array)  #Add to the lists
        labels.append(row['label'])
    X = np.array(images)  #Convert to numpy arrays
    y = 2 * np.array(labels) - 1  #and convert labels to the format {-1, 1}, to be used by EstimatorQNN
    return X, y

# ========== QUANTUM CIRCUIT HELPERS ==========
def control_positions(row, col, img_size):  #Function to get control positions for FRQI
    num_bits = int(np.log2(img_size))  #Calculate the number of bits needed for row and column
    row_binary = format(row, f'0{num_bits}b')
    col_binary = format(col, f'0{num_bits}b')
    qubit_string = row_binary + col_binary
    return [pos for pos, x in enumerate(reversed(qubit_string)) if x == '0']

def multi_controlled_ry_gray_code(qc, controls, target, theta):  #Function to decompose multi-controlled Ry, using gray code method
    n_controls = len(controls)
    n_rotations = 2**n_controls
    for i in range(n_rotations):  #Apply Ry rotations with gray code CNOT sequence
        angle = theta / n_rotations  #Calculate rotation angle for this step
        qc.ry(angle, target)  #Apply Ry rotation
        if i < n_rotations - 1:  #Apply CNOT based on gray code if not last iteration
            gray_current = i ^ (i >> 1)  #Find which bit changes in gray code sequence
            gray_next = (i + 1) ^ ((i + 1) >> 1)
            changed_bit = gray_current ^ gray_next
            control_qubit = 0  #Find position of changed bit
            while (changed_bit >> control_qubit) & 1 == 0:
                control_qubit += 1
            qc.cx(controls[control_qubit], target)  #Apply CNOT

def multi_controlled_ry_recursive(qc, controls, target, theta):  #Function to recursively decompose multi-controlled Ry
    n_controls = len(controls)    
    if n_controls == 0:
        qc.ry(theta, target)
    elif n_controls == 1:
        qc.cry(theta, controls[0], target)
    elif n_controls == 2:  #Use the 2-controlled pattern
        qc.cry(theta/2, controls[1], target)
        qc.cx(controls[0], controls[1])
        qc.cry(-theta/2, controls[1], target)
        qc.cx(controls[0], controls[1])
        qc.cry(theta/2, controls[0], target)
    else:  #Split controls in half and apply recursively
        mid = n_controls // 2
        controls1 = controls[:mid]
        controls2 = controls[mid:]
        #This requires an ancilla qubit approach or further decomposition
        #For simplicity, use the iterative gray code method above
        multi_controlled_ry_gray_code(qc, controls, target, theta)

def create_frqi_circuit(img_size):  #Function to create a parameterized FRQI circuit for images of size img_size*img_size
    num_pos_qubits = 2 * int(np.log2(img_size)) #Calculate dimensions
    qpos = QuantumRegister(num_pos_qubits, 'p')  #Create quantum registers and circuit
    qcolor = QuantumRegister(1, 'c')
    qc = QuantumCircuit(qpos, qcolor)
    qc.h(qpos)
    phi_params = {}  #Create parameters for each pixel
    for i in range(img_size):  #For each pixel, apply a parameterized rotation
        for j in range(img_size):
            control_indices = control_positions(i, j, img_size)
            phi_params[i, j] = Parameter(f"phi_{i}_{j}")
            if control_indices:  #Apply X gates to set up the control pattern
                qc.x(control_indices)
            multi_controlled_ry_recursive(qc, list(range(num_pos_qubits)), qcolor[0], 2 * np.pi * phi_params[i, j])  #Apply the decomposed multi-controlled Ry gate
            if control_indices:  #Undo the X gates
                qc.x(control_indices)
    return qc, phi_params

# ========== MODEL ==========
class HybridModel(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn = qnn
        self.classification_layer = nn.Linear(5, 1)  #5 qubits in total

    def forward(self, x):
        x = self.qnn(x)
        x = self.classification_layer(x)
        return x

# ========== RESOURCE USAGE HELPERS ==========
def get_quantum_metrics(qc):
    return {
        'qubit_count': qc.num_qubits,
        'circuit_depth': qc.depth(),
        'gate_count': dict(qc.count_ops())
    }

def get_classical_metrics(model):
    params = sum(p.numel() for p in model.parameters())
    return {
        'parameter_count': params,
        'estimated_memory_footprint_bytes': int(params * 4)
    }

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(description='QNN Training and Evaluation')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with labels.csv and images')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #Output directory setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # ========== DATA ==========
    start_data = time.time() ##TIME TRACKING
    labels_file = os.path.join(args.data_dir, 'labels.csv')
    X, y = load_image_dataset(labels_file, IMAGE_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_time = time.time() - start_data ##TIME TRACKING 

    # ========== QNN ==========
    start_qnn = time.time() ##TIME TRACKING
    
    start_frqi = time.time() ##TIME TRACKING
    frqi_circuit, phi_params = create_frqi_circuit(IMAGE_SIZE)
    frqi_time = time.time() - start_frqi ##TIME TRACKING
    
    start_ansatz = time.time() ##TIME TRACKING
    ansatz = EfficientSU2(num_qubits=frqi_circuit.num_qubits, reps=2, entanglement='linear')
    ansatz_time = time.time() - start_ansatz ##TIME TRACKING
    
    start_compose = time.time() ##TIME TRACKING
    qnn_circuit = frqi_circuit.compose(ansatz)
    compose_time = time.time() - start_compose ##TIME TRACKING
    
    start_qnn_init = time.time() ##TIME TRACKING
    observables = [
      SparsePauliOp.from_list([("IIIIZ", 1)]),
      SparsePauliOp.from_list([("IIIZI", 1)]),
      SparsePauliOp.from_list([("IIZII", 1)]),
      SparsePauliOp.from_list([("IZIII", 1)]),
      SparsePauliOp.from_list([("ZIIII", 1)])
    ]  #Define the observables
    initial_ansatz_weights = [2 * torch.rand(1).item() - 1 for _ in range(ansatz.num_parameters)]  #Define initial guesses for all the phi parameters
    estimator = StatevectorEstimator()  #Create the EstimatorQNN
    qnn = EstimatorQNN(
      circuit=qnn_circuit,
      observables=observables,
      input_params=list(phi_params.values()),  #The input parameters are the phi_params of the FRQI
      weight_params=ansatz.parameters,  #The weight parameters are from the ansatz
      input_gradients=True,
      estimator=estimator
    )
    qnn_init_time = time.time() - start_qnn_init ##TIME TRACKING

    start_torch = time.time() ##TIME TRACKING
    qnn_torch = TorchConnector(qnn, initial_weights=initial_ansatz_weights)  #Do the connection between QNN and PyTorch
    model = HybridModel(qnn_torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.SoftMarginLoss()
    torch_time = time.time() - start_torch ##TIME TRACKING
    
    qnn_time = time.time() - start_qnn ##TIME TRACKING

    # ========== TRAIN ==========
    print("Training...")
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    loss_history = []
    epoch_times = [] ##TIME TRACKING
    batch_times = [] ##TIME TRACKING
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time() ##TIME TRACKING
        epoch_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_start = time.time() ##TIME TRACKING
            optimizer.zero_grad()
            
            start_forward = time.time() ##TIME TRACKING
            output = model(data)
            forward_time = time.time() - start_forward ##TIME TRACKING
            
            start_loss = time.time() ##TIME TRACKING
            loss = loss_func(output, target.unsqueeze(1))
            loss_time = time.time() - start_loss ##TIME TRACKING
            
            start_backward = time.time() ##TIME TRACKING
            loss.backward()
            backward_time = time.time() - start_backward ##TIME TRACKING
            
            start_step = time.time() ##TIME TRACKING
            optimizer.step()
            step_time = time.time() - start_step ##TIME TRACKING
            
            batch_time = time.time() - batch_start ##TIME TRACKING
            batch_times.append({
                'epoch': epoch,
                'batch': batch_time,
                'forward': forward_time,
                'loss': loss_time,
                'backward': backward_time,
                'step': step_time,
            })
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start ##TIME TRACKING
        epoch_times.append(epoch_time) ##TIME TRACKING
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    training_time = time.time() - start_time
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem

    # ========== SAVE TRAINING CURVE ==========
    plt.figure()
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # ========== EVALUATION ==========
    print("Evaluating...")
    model.eval()
    inference_start = time.time()
    with torch.no_grad():
        y_pred = model(torch.Tensor(X_test)).squeeze()
        y_probs = torch.sigmoid(y_pred).numpy()
        y_pred_labels = torch.sign(y_pred).numpy()
    inference_time = (time.time() - inference_start) / len(X_test)

    #Convert labels for metrics
    y_test_binary = (y_test + 1) // 2
    y_pred_binary = (y_pred_labels + 1) // 2

    #Metrics
    accuracy = (y_pred_binary == y_test_binary).mean()
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    specificity = recall_score(1 - y_test_binary, 1 - y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    fpr = 1 - specificity
    auc_score = roc_auc_score(y_test_binary, y_probs)
    fpr_curve, tpr_curve, _ = roc_curve(y_test_binary, y_probs)
    cm = confusion_matrix(y_test_binary, y_pred_binary)

    #Resource metrics
    quantum_metrics = get_quantum_metrics(qnn_circuit)
    classical_metrics = get_classical_metrics(model)

    #Energy estimation (simple placeholder)
    quantum_energy = quantum_metrics['qubit_count'] * quantum_metrics['circuit_depth'] * 1e-8
    classical_energy = classical_metrics['parameter_count'] * 1e-9
    total_energy = quantum_energy + classical_energy

    # ========== SAVE METRICS ==========
    metrics = {
        'accuracy': float(accuracy),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr),
        'auc_roc': float(auc_score),
        'training_time_sec': float(training_time),
        'inference_time_per_sample_sec': float(inference_time),
        'memory_usage_kb': int(mem_usage),
        'quantum_metrics': quantum_metrics,
        'classical_metrics': classical_metrics,
        'energy_estimate_joules': {
            'quantum': quantum_energy,
            'classical': classical_energy,
            'total': total_energy
        },
        'scalability_notes': {
            'qubit_scaling': f"O(log(n)) for {IMAGE_SIZE}x{IMAGE_SIZE} images",
            'parameter_scaling': "Linear with image size"
        },
        'data_loading_time_sec': float(data_time),
        'qnn_setup_time_sec': float(qnn_time),
        'qnn_setup_breakdown_sec': {
            'frqi_circuit': frqi_time,
            'ansatz': ansatz_time,
            'compose': compose_time,
            'qnn_init': qnn_init_time,
            'torch_connector': torch_time
        },
        'training_time_sec': float(sum(epoch_times)),
        'training_time_per_epoch_sec': [float(t) for t in epoch_times],
        'training_time_breakdown_per_batch': batch_times,
    }
    with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ========== SAVE REPORTS & PLOTS ==========
    #Classification report
    report = classification_report(y_test_binary, y_pred_binary, target_names=['No Ellipse', 'Ellipse'])
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    #Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Ellipse', 'Ellipse'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (FPR: {fpr:.2f})')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    #ROC curve
    plt.figure()
    plt.plot(fpr_curve, tpr_curve, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    #Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
    np.save(os.path.join(output_dir, 'test_predictions.npy'), y_pred_labels)

    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    main()
