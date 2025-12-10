#!/usr/bin/env python3
"""
Train FRQI + Variational Quantum Classifier using PennyLane + PyTorch.
Saves metrics, plots, circuit statistics, predictions, and weights.
"""
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # << add this
import matplotlib.pyplot as plt
from tqdm import tqdm
import resource

# Torch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)

# PennyLane
import pennylane as qml
from pennylane import numpy as pnp  # use pnp if needed for PL-native arrays

# Use PL numpy also as np for compatibility in the FRQI function
from pennylane import numpy as np_pl

# ------------------------- Data loading ----------------------------

def load_image_dataset(labels_file, img_size=4, sample_size=None):
    df = pd.read_csv(labels_file)
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    images = []
    labels = []
    for _, row in df.iterrows():
        img = Image.open(row["filename"]).convert("L")
        # ensure correct size - if dataset already 4x4 this is idempotent
        img = img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        images.append(arr.flatten())
        labels.append(row["label"])

    X = np.array(images)              # shape [N, img_size^2]
    y = 2*np.array(labels)-1          # map {0,1} -> {-1,1}
    return X, y, df

# ------------------------- FRQI state builder ----------------------

def build_frqi_statevector(image):
    """
    Build FRQI amplitude vector for a single image (4x4).
    Expects `image` as a 2D or flattened numpy array of floats in [0,1].
    Produces a real statevector of length 2 * (4*4) = 32 and normalizes it.
    State ordering: [cos(x_00), sin(x_00), cos(x_01), sin(x_01), ...]
    """
    flat = np.array(image).flatten()
    state = []
    for x in flat:
        state.append(np.cos(np.pi * x))
        state.append(np.sin(np.pi * x))
    state = np.array(state, dtype=float)
    # Normalize
    state = state / np.linalg.norm(state)
    return state

# ------------------------- Build QNode ------------------------------

def build_qnode(num_layers=5):
    num_pos_qubits = 4
    num_color_qubit = 1
    num_wires = num_pos_qubits + num_color_qubit
    color_wire = num_pos_qubits

    dev = qml.device("default.qubit", wires=num_wires)

    OBS = [qml.PauliZ(i) for i in range(num_wires)]

    @qml.qnode(dev, interface="torch")
    def qnode_circuit(inputs, weights):
        # inputs: torch tensor or array of length img_size^2 (16)
        # reshape and build FRQI amplitudes using numpy/pennylane numpy
        # Convert to numpy array (PennyLane will convert torch -> numpy-like automatically)
        img = np_pl.array(inputs).reshape(4, 4)

        # build FRQI statevector
        # Using the same builder as before
        state = build_frqi_statevector(img)
        # Use StatePrep to prepare the multi-qubit state
        qml.StatePrep(state, wires=range(num_wires))

        # Variational Ansatz
        qml.BasicEntanglerLayers(weights, wires=range(num_wires))

        # return expectation values for each qubit's Z
        return [qml.expval(o) for o in OBS]

    weight_shapes = {"weights": (num_layers, num_wires)}
    return qnode_circuit, weight_shapes, num_wires

# -------------------- Trainable observable layer -------------------

class TrainableObsLayer(nn.Module):
    def __init__(self, num_obs):
        super().__init__()
        # initialize small weights for stability
        self.obs_weights = nn.Parameter(torch.randn(num_obs) * 0.1)

    def forward(self, obs_vec):
        # obs_vec is a torch tensor of expectation values
        return torch.dot(self.obs_weights, obs_vec)

# ----------------------- Full Hybrid Model ------------------------

class HybridFRQI(nn.Module):
    def __init__(self, qnode, weight_shapes, num_obs):
        super().__init__()
        self.qnn = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.obs = TrainableObsLayer(num_obs)

    def forward(self, x):
        obs_vec = self.qnn(x)
        return self.obs(obs_vec)

# ----------------------- Utility: draw circuit --------------------

def save_circuit_diagram(qnode, example_input, example_weights, out_path):
    try:
        qml.drawer.use_style("pennylane_sketch")
        fig = qml.draw_mpl(qnode)(example_input, example_weights)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Could not draw circuit diagram:", e)

# --------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing labels.csv and images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--sample-size", type=int, default=None, help="If set, randomly sample this many images from labels.csv")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of BasicEntanglerLayers")
    args = parser.parse_args()

    IMG_SIZE = 4

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ------------------- Load Dataset -------------------
    t0 = time.time()
    labels_file = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"labels.csv not found in {args.data_dir}")

    X, y, df_used = load_image_dataset(labels_file, IMG_SIZE, sample_size=args.sample_size)
    N = X.shape[0]
    data_time = time.time() - t0
    print(f"Loaded {N} samples in {data_time:.2f}s")

    # Save a copy of CSV used
    df_used.to_csv(os.path.join(out_dir, "labels_used.csv"), index=False)

    # ------------------- Build QNode ---------------------
    qnode, weight_shapes, num_wires = build_qnode(num_layers=args.num_layers)

    # Example items for drawing
    example_input = torch.tensor(X[0]).float()
    dummy_weights = torch.zeros(tuple(weight_shapes["weights"]))

    # Save circuit diagram (best-effort)
    try:
        save_circuit_diagram(qnode, example_input, dummy_weights, os.path.join(out_dir, "circuit.png"))
        print("Saved circuit diagram to circuit.png")
    except Exception as e:
        print("Failed to draw circuit:", e)

    # Create dataset and dataloader
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = HybridFRQI(qnode, weight_shapes, num_obs=num_wires)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SoftMarginLoss()

    # ------------------- TRAIN ---------------------------
    loss_history = []
    model.train()

    train_start = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for epoch in range(args.epochs):
        batch_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            optimizer.zero_grad()

            # xb is [batch_size, 16]; TorchLayer expects individual inputs,
            # so we loop over batch (this is how you had it originally).
            preds = [model(x) for x in xb]
            preds = torch.stack(preds).squeeze()

            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            pbar.set_postfix({"loss": float(np.mean(batch_losses))})

        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: loss = {epoch_loss:.4f}")

    train_time = time.time() - train_start
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem

    # ------------------- SAVE TRAINING CURVE --------------
    plt.figure(figsize=(7,4))
    plt.plot(loss_history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # ------------------- SAVE LOSS HISTORY AS CSV --------------
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(loss_history) + 1)),
        "loss": loss_history
    })
    loss_csv_path = os.path.join(out_dir, "loss_history.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Saved loss history to {loss_csv_path}")

    # ------------------- EVALUATION -----------------------
    model.eval()
    with torch.no_grad():
        preds = []
        for x in X:
            preds.append(model(torch.tensor(x).float()).item())

    preds = np.array(preds)
    pred_labels = np.sign(preds)

    acc = (pred_labels == y).mean()

    # Convert to binary {0,1}
    y_bin = (y + 1) // 2
    pred_bin = (pred_labels + 1) // 2

    # handle degenerate/constant cases for auc
    try:
        auc = roc_auc_score(y_bin, preds)
    except Exception:
        auc = float("nan")

    precision = precision_score(y_bin, pred_bin, zero_division=0)
    recall = recall_score(y_bin, pred_bin, zero_division=0)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    cm = confusion_matrix(y_bin, pred_bin)
    try:
        fpr_curve, tpr_curve, _ = roc_curve(y_bin, preds)
    except Exception:
        fpr_curve, tpr_curve = [0,1], [0,1]

    # ---------------- SAVE CONFUSION MATRIX ----------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # --------------------- SAVE ROC ------------------------
    plt.figure(figsize=(6,6))
    plt.plot(fpr_curve, tpr_curve, label=f"AUC = {auc:.2f}" if not np.isnan(auc) else "AUC = n/a")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # ------------------- SAVE METRICS ----------------------
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc) if not np.isnan(auc) else None,
        "training_time_sec": float(train_time),
        "memory_usage_kb": int(mem_usage),
        "data_loading_time_sec": float(data_time),
        "num_qubits": int(num_wires),
        "num_samples": int(N),
        "num_layers": int(args.num_layers),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------- SAVE WEIGHTS + PREDS -------------- 
    # Also save a small CSV with filename, true label, predicted score, predicted label
    results_df = pd.DataFrame({
        "filename": df_used["filename"],
        "true_label": df_used["label"],
        "pred_score": preds,
        "pred_label": ((pred_labels + 1) // 2).astype(int),
    })
    results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print("\nAll results saved in:", out_dir)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
