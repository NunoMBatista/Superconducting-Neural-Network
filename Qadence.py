import os
import time
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from qadence import (
    chain, kron, QuantumCircuit, run,
    H, X, MCRY, Z, RY, RX, CNOT,
    QNN, VariationalParameter, FeatureParameter, hea
)
import torch
from torch.optim import Adam
import seaborn as sns
import argparse

# === Define Command-Line Arguments ===
parser = argparse.ArgumentParser(description="Run circuit training with configurable parameters.")
parser.add_argument('--image_size', type=int, default=4, help='Size of input images (e.g., 28 for 28x28)')
parser.add_argument('--depth', type=int, default=3, help='Depth of the circuit')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')

args = parser.parse_args()

IMAGE_SIZE = args.image_size
depth = args.depth
n_epochs = args.epochs
RESULTS_DIR = args.results_dir



# === Output directory ===
#RESULTS_DIR = "/home/amorgado/FRQI_QNN/results/extra_qadence/image4_depth03_epoch9000"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Constants ===
#IMAGE_SIZE = 4
NUM_IMAGES = 512
GRAYSCALE_LEVELS = 8

# === Data generation functions ===
def create_smooth_gradient_background(size):
    corners = np.random.randint(0, GRAYSCALE_LEVELS, size=4) * (255 // (GRAYSCALE_LEVELS - 1))
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    top = corners[0] * (1 - x) + corners[1] * x
    bottom = corners[2] * (1 - x) + corners[3] * x
    gradient = top * (1 - y) + bottom * y
    return np.round(gradient).astype(np.uint8)

def add_ellipse(img_array, size):
    temp_img = Image.fromarray(img_array, 'L')
    draw = ImageDraw.Draw(temp_img)
    x0, y0 = random.randint(0, size-3), random.randint(0, size-3)
    x1, y1 = x0 + random.randint(2, size//2), y0 + random.randint(2, size//2)
    fill_value = random.randint(0, GRAYSCALE_LEVELS-1) * (255 // (GRAYSCALE_LEVELS - 1))
    draw.ellipse([x0, y0, x1, y1], fill=fill_value)
    return np.array(temp_img)

def create_image(has_ellipse):
    img_array = create_smooth_gradient_background(IMAGE_SIZE)
    if has_ellipse:
        img_array = add_ellipse(img_array, IMAGE_SIZE)
    return img_array

# === Generate training data ===
dataset = []
for i in tqdm(range(NUM_IMAGES), desc="Generating training data"):
    has_ellipse = random.random() < 0.5
    img_array = create_image(has_ellipse)
    dataset.append({'image': img_array, 'label': has_ellipse})
df_train = pd.DataFrame(dataset)

# === Quantum circuit setup ===
feature_params = {
    (row, col): FeatureParameter(f'x{row}{col}')
    for row in range(IMAGE_SIZE)
    for col in range(IMAGE_SIZE)
}
feature_params_list = list(feature_params.values())
qdim = math.floor(math.log2(IMAGE_SIZE))
n_qubits = 2 * qdim + 1
control_qubits = list(range(n_qubits-1))

def control_state(row: int, col: int):
    row_binary = f"{row:b}".rjust(qdim, '0')
    col_binary = f"{col:b}".rjust(qdim, '0')
    return row_binary + col_binary

ops_feature = [kron(H(i) for i in range(n_qubits - 1))]
for (row, col), parameter in feature_params.items():
    rotation_gate = kron(MCRY(control_qubits, n_qubits - 1, parameter * np.pi))
    cstate = control_state(row, col)
    qubits_to_flip = [i for i, x in enumerate(cstate) if x == '1']
    if qubits_to_flip:
        ops_feature.append(kron(X(i) for i in qubits_to_flip))
    ops_feature.append(rotation_gate)
    if qubits_to_flip:
        ops_feature.append(kron(X(i) for i in qubits_to_flip))
chain_feature = chain(*ops_feature)
qc_feature = QuantumCircuit(n_qubits, chain_feature)
#depth = 3
qc_ansatz = hea(n_qubits, depth)
obs_parameters = [VariationalParameter(f'z{i}') for i in range(n_qubits)]
observable = sum(obs_parameters[i] * Z(i) for i in range(n_qubits)) / n_qubits
ops_all = chain(*qc_feature, qc_ansatz)
qc = QuantumCircuit(n_qubits, ops_all)
model = QNN(qc, observable, inputs=feature_params_list)

# === Prepare training data ===
img_train = df_train.image / 255 * np.pi
y_train = torch.tensor(np.array(df_train.label), dtype=torch.float64) * 2 - 1
input_dict_train = {
    feature.name: torch.tensor([img[i, j] for img in img_train], dtype=torch.float64)
    for (i, j), feature in feature_params.items()
}

# === Training ===
criterion = torch.nn.SoftMarginLoss()
def loss_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model_output = model.expectation(values=input_dict_train)
    loss = criterion(model_output.squeeze(), y_train)
    return loss

epoch_losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#n_epochs = 9000

start_time = time.time()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn(model, input_dict_train)
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print(f"{epoch=}, Loss: {loss.item():.4f}", flush=True) #To force the print to the document every time
    epoch_losses.append({'epoch': epoch, 'loss': loss.detach().cpu().numpy()})
training_time = time.time() - start_time

# === Save loss curve ===
loss_df = pd.DataFrame(epoch_losses)
loss_csv_path = os.path.join(RESULTS_DIR, "loss_history.csv")
loss_df.to_csv(loss_csv_path, index=False)

plt.figure()
plt.plot(loss_df['epoch'], loss_df['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid()
loss_img_path = os.path.join(RESULTS_DIR, "loss_curve.png")
plt.savefig(loss_img_path)
plt.close()

# === Generate test data ===
dataset_test = []
for i in tqdm(range(NUM_IMAGES), desc="Generating test data"):
    has_ellipse = random.random() < 0.5
    img_array = create_image(has_ellipse)
    dataset_test.append({'image': img_array, 'label': has_ellipse})
df_test = pd.DataFrame(dataset_test)

img_test = df_test.image / 255 * np.pi
y_test = torch.tensor(np.array(df_test.label), dtype=torch.float64) * 2 - 1
input_dict_test = {
    feature.name: torch.tensor([img[i, j] for img in img_test], dtype=torch.float64)
    for (i, j), feature in feature_params.items()
}

# === Evaluation ===
eval_start_time = time.time()
y_pred = model.expectation(values=input_dict_test).squeeze()
eval_time = time.time() - eval_start_time
total_time = training_time + eval_time

def classify(circuit_output):
    return int(bool(circuit_output > 0))
test_values = np.vectorize(classify)(y_test.detach().cpu().numpy())
predicted_values = np.vectorize(classify)(y_pred.detach().cpu().numpy())
confusion = np.zeros((2, 2), dtype=int)
for pred, true in zip(predicted_values, test_values):
    confusion[pred, true] += 1
    
# Calculate accuracy
num_correct = np.trace(confusion)
num_total = np.sum(confusion)
accuracy = num_correct / num_total
print(f"Accuracy: {accuracy:.4f}")

# === Save confusion matrix as image ===
plt.figure()
sns.heatmap(confusion / NUM_IMAGES, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.xticks([0.5, 1.5], ["Real\nNegative", "Real\nPositive"])
plt.yticks([0.5, 1.5], ["Predicted\nNegative", "Predicted\nPositive"])
plt.title('Confusion Matrix')
conf_matrix_img = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_img)
plt.close()

# === Save metrics ===
false_negative = confusion[0, 1] / np.sum(confusion[0])
false_positive = confusion[1, 0] / np.sum(confusion[1])
metrics = {
    "false_negative": false_negative,
    "false_positive": false_positive,
    "accuracy": accuracy,
    "training_time_sec": training_time,
    "evaluation_time_sec": eval_time,
    "total_time_sec": total_time
}
metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

# === Save confusion matrix as CSV ===
confusion_df = pd.DataFrame(confusion, columns=["True Negative", "True Positive"], index=["Predicted Negative", "Predicted Positive"])
confusion_csv_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
confusion_df.to_csv(confusion_csv_path)

print("All results saved to:", RESULTS_DIR)
