"""
Train FRQI + Variational Quantum Classifier using PennyLane + JAX.
Saves metrics, plots, circuit statistics, predictions, and weights.
Backend: default.qubit (Accelerated via JAX/XLA on GPU).
Saves metrics, plots, circuit statistics, predictions, and weights.
Supports arbitrary square image sizes (e.g. 4x4, 16x16)


Run as:
    python3 03b_Pennylane_FRQIStateVector_QNN_lightningDevice_JAX.py --data-dir ./Datasets/dataset_250_4by4 --img-size 4 --epochs 20 --batch-size 16 --output-dir ./Code02_epochs20
"""

import os
import warnings
warnings.filterwarnings("ignore", message="PennyLane is currently not compatible with versions of JAX")

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
import pickle

# JAX & Optax
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
from functools import partial

# PennyLane
import pennylane as qml

# Metrics
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)

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

    X = jnp.array(images)
    y = 2 * jnp.array(labels) - 1  # {0,1} → {-1,1}

    return X, y, df

# ------------------------- FRQI state builder ----------------------

def build_frqi_statevector(image):
    flat = image.flatten()
    cos_vals = jnp.cos(jnp.pi * flat)
    sin_vals = jnp.sin(jnp.pi * flat)
    state = jnp.stack([cos_vals, sin_vals], axis=-1).reshape(-1)
    return state / jnp.linalg.norm(state)

# ------------------------- Build QNode ------------------------------

def build_qnode(img_size, num_layers):
    num_pixels = img_size * img_size
    num_pos_qubits = int(np.log2(num_pixels))
    num_wires = num_pos_qubits + 1

    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev, interface="jax")
    def qnode(inputs, weights):
        state = build_frqi_statevector(inputs)
        qml.StatePrep(state, wires=range(num_wires))
        qml.BasicEntanglerLayers(weights, wires=range(num_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

    weight_shape = (num_layers, num_wires)
    return qnode, weight_shape, num_wires

# ----------------------- Full Hybrid Model ------------------------

def model_forward(params, x, qnode):
    # FIXED: Wrap the qnode output list in jnp.array for JAX compatibility
    obs_vec = jnp.array(qnode(x, params["q_weights"]))
    return jnp.dot(params["obs_weights"], obs_vec)

# ----------------------- Loss & Training --------------------------

def soft_margin_loss(params, x_batch, y_batch, qnode):
    preds = vmap(lambda x: model_forward(params, x, qnode))(x_batch)
    return jnp.mean(jnp.logaddexp(0.0, -y_batch * preds))

@partial(jit, static_argnames=("qnode", "optimizer"))
def train_step(params, opt_state, x_batch, y_batch, qnode, optimizer):
    loss, grads = value_and_grad(soft_margin_loss)(params, x_batch, y_batch, qnode)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ----------------------- Evaluation helper -------------------------

def evaluate_model(params, X, y, qnode):
    preds = vmap(lambda x: model_forward(params, x, qnode))(X)
    preds = np.array(preds)
    pred_labels = np.sign(preds)

    acc = (pred_labels == np.array(y)).mean()

    y_bin = (np.array(y) + 1) // 2
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
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    X_train, y_train, df_train = load_image_dataset(os.path.join(args.data_dir, "labels_train.csv"), args.img_size, args.sample_size)
    X_val, y_val, _ = load_image_dataset(os.path.join(args.data_dir, "labels_val.csv"), args.img_size)
    X_test, y_test, df_test = load_image_dataset(os.path.join(args.data_dir, "labels_test.csv"), args.img_size)

    qnode, weight_shape, num_wires = build_qnode(args.img_size, args.num_layers)

    key = jax.random.PRNGKey(42)
    params = {
        "q_weights": 0.01 * jax.random.normal(key, weight_shape),
        "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))
    }

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    loss_history, metrics_history = [], []
    eval_interval = 10
    metrics_csv = os.path.join(out_dir, "metrics_by_epoch.csv")

    train_start = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for epoch in range(args.epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, len(X_train))
        X_shuff, y_shuff = X_train[idx], y_train[idx]

        batch_losses = []
        pbar = tqdm(range(0, len(X_train), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for i in pbar:
            xb, yb = X_shuff[i : i + args.batch_size], y_shuff[i : i + args.batch_size]
            params, opt_state, loss = train_step(params, opt_state, xb, yb, qnode, optimizer)
            batch_losses.append(float(loss))
            pbar.set_postfix({"loss": float(np.mean(batch_losses))})

        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: loss = {epoch_loss:.4f}")

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == args.epochs:
            acc, prec, rec, f1, auc = evaluate_model(params, X_val, y_val, qnode)
            metrics_history.append({
                "epoch": epoch + 1, "loss": epoch_loss, "accuracy": acc,
                "precision": prec, "recall": rec, "f1_score": f1, "auc": auc
            })
            pd.DataFrame(metrics_history).to_csv(metrics_csv, index=False)
            print(f"  Eval → acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

    train_time = time.time() - train_start
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem

    # Final Eval
    test_preds = vmap(lambda x: model_forward(params, x, qnode))(X_test)
    test_preds = np.array(test_preds)
    pred_labels, y_test_np = np.sign(test_preds), np.array(y_test)
    y_bin, pred_bin = (y_test_np + 1) // 2, (pred_labels + 1) // 2

    acc = (pred_labels == y_test_np).mean()
    prec = precision_score(y_bin, pred_bin, zero_division=0)
    rec = recall_score(y_bin, pred_bin, zero_division=0)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_bin, test_preds)
        fpr, tpr, _ = roc_curve(y_bin, test_preds)
    except:
        auc, fpr, tpr = float("nan"), [0, 1], [0, 1]

    # Saving Results
    pd.DataFrame({"epoch": range(1, len(loss_history)+1), "loss": loss_history}).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
    
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1_score": float(f1), "auc": float(auc) if not np.isnan(auc) else None,
        "training_time_sec": float(train_time), "memory_usage_kb": int(mem_usage),
        "data_loading_time_sec": float(time.time() - t0), "num_qubits": int(num_wires),
        "num_samples_test": int(len(X_test)), "num_layers": int(args.num_layers),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "filename": df_test["filename"], "true_label": df_test["label"],
        "pred_score": test_preds, "pred_label": pred_bin.astype(int),
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    plt.figure(figsize=(7,4)); plt.plot(loss_history); plt.title("Loss"); plt.savefig(os.path.join(out_dir, "training_loss.png")); plt.close()
    
    cm = confusion_matrix(y_bin, pred_bin)
    cm_df = pd.DataFrame(
        cm,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"]
    )
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues"); plt.savefig(os.path.join(out_dir, "confusion_matrix.png")); plt.close()

    plt.figure(figsize=(6,6)); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig(os.path.join(out_dir, "roc_curve.png")); plt.close()

    with open(os.path.join(out_dir, "model_weights.pkl"), "wb") as f:
        pickle.dump(params, f)

    print(f"\nAll results saved in: {out_dir}")

if __name__ == "__main__":
    main()