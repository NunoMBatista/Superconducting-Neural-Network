"""
Train FRQI + Variational Quantum Classifier using PennyLane + PyTorch.
Saves metrics, plots, circuit statistics, predictions, and weights.
Supports arbitrary square image sizes (e.g. 4x4, 16x16)

Run as:
    python3 Pennylane_FRQIStateVector_QNN.py --data-dir ./Datasets/dataset_250_4by4 --img-size 4 --epochs 20 --batch-size 16 --output-dir ./Code02_epochs20
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg") 
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
        img = img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0

        images.append(arr.flatten())
        labels.append(row["label"])

    X = np.array(images)
    y = 2 * np.array(labels) - 1  # {0,1} → {-1,1}

    return X, y, df

# ------------------------- FRQI state builder ----------------------

def build_frqi_statevector(image):
    """
    Build FRQI amplitude vector for a single square image.
    Statevector length = 2 * (img_size^2)
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

def build_qnode(img_size, num_layers):
    num_pixels = img_size * img_size
    num_pos_qubits = int(np.log2(num_pixels))
    num_color_qubit = 1
    num_wires = num_pos_qubits + num_color_qubit

    dev = qml.device("lightning.qubit", wires=num_wires)
    OBS = [qml.PauliZ(i) for i in range(num_wires)]

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        img = np_pl.array(inputs).reshape(img_size, img_size)
        state = build_frqi_statevector(img)
        qml.StatePrep(state, wires=range(num_wires))
        qml.BasicEntanglerLayers(weights, wires=range(num_wires))
        return [qml.expval(o) for o in OBS]

    weight_shapes = {"weights": (num_layers, num_wires)}
    return qnode, weight_shapes, num_wires

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

# ----------------------- Evaluation helper -------------------------

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = [model(torch.tensor(x).float()).item() for x in X]

    preds = np.array(preds)
    pred_labels = np.sign(preds)

    acc = (pred_labels == y).mean()

    y_bin = (y + 1) // 2
    pred_bin = (pred_labels + 1) // 2

    precision = precision_score(y_bin, pred_bin, zero_division=0)
    recall = recall_score(y_bin, pred_bin, zero_division=0)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)

    try:
        auc = roc_auc_score(y_bin, preds)
    except Exception:
        auc = float("nan")

    return acc, precision, recall, f1, auc

# --------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing labels.csv and images")
    parser.add_argument("--img-size", type=int, required=True, help="Image width/height (e.g. 4 or 16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--sample-size", type=int, default=None, help="If set, randomly sample this many images from labels.csv")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of BasicEntanglerLayers")
    args = parser.parse_args()

    IMG_SIZE = args.img_size

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ------------------- Load datasets -------------------
    t0 = time.time()

    X_train, y_train, df_train = load_image_dataset(
        os.path.join(args.data_dir, "labels_train.csv"),
        args.img_size,
        args.sample_size,
    )
    X_val, y_val, _ = load_image_dataset(
        os.path.join(args.data_dir, "labels_val.csv"),
        args.img_size,
    )
    X_test, y_test, df_test = load_image_dataset(
        os.path.join(args.data_dir, "labels_test.csv"),
        args.img_size,
    )

    print(f"Loaded train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ------------------- Build QNode ---------------------
    qnode, weight_shapes, num_wires = build_qnode(args.img_size, args.num_layers)

    model = HybridFRQI(qnode, weight_shapes, num_obs=num_wires)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SoftMarginLoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(y_train).float()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # ------------------- TRAIN ---------------------------
    loss_history = []
    metrics_history = []
    eval_interval = 10
    metrics_csv = os.path.join(out_dir, "metrics_by_epoch.csv")
    model.train()

    train_start = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
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

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == args.epochs:
            acc, prec, rec, f1, auc = evaluate_model(model, X_val, y_val)

            metrics_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc": auc
            })

            pd.DataFrame(metrics_history).to_csv(metrics_csv, index=False)

            print(
                f"  Eval → acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}"
            )

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

    # ------------------- FINAL EVALUATION (TEST SET) -----------------------
    model.eval()
    with torch.no_grad():
        preds = []
        for x in X_test:
            preds.append(model(torch.tensor(x).float()).item())

    preds = np.array(preds)
    pred_labels = np.sign(preds)

    acc = (pred_labels == y_test).mean()

    # Convert to binary {0,1}
    y_bin = (y_test + 1) // 2
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
        fpr_curve, tpr_curve = [0, 1], [0, 1]

    # ---------------- SAVE CONFUSION MATRIX ----------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    cm_df = pd.DataFrame(
        cm,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"]
    )
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # --------------------- SAVE ROC ------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr_curve,
        tpr_curve,
        label=f"AUC = {auc:.2f}" if not np.isnan(auc) else "AUC = n/a"
    )
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # ------------------- SAVE METRICS ----------------------
    data_time = time.time() - t0

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
        "num_samples_test": int(len(X_test)),
        "num_layers": int(args.num_layers),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------- SAVE PREDICTIONS ------------------
    results_df = pd.DataFrame({
        "filename": df_test["filename"],
        "true_label": df_test["label"],
        "pred_score": preds,
        "pred_label": ((pred_labels + 1) // 2).astype(int),
    })
    results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print("\nAll test results saved in:", out_dir)
    print("Test metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
