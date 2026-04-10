"""
K-Fold Cross-Validation with Early Stopping for QML.
Uses your specific FRQI logic and evaluation structure.

Run as:
    python3 05_code03b_kfoldcrossvalidation.py --data-dir ./Datasets/dataset_250_16by16 --img-size 16 --k-folds 5 --max-epochs 500 --patience 20 --batch-size 16 --lr 0.02 --num-layers 7 --output-dir ./Code02_epochs20
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" #forces JAX to not reserve about 90% of VRAM at the beginning
import warnings
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
import pennylane as qml
from functools import partial
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
)

warnings.filterwarnings("ignore", message="PennyLane is currently not compatible with versions of JAX")

# ------------------------- Data loading (Full Dataset for K-Fold) ----------------------------

def load_full_dataset(data_dir, img_size):
    """Combines all labels files to perform a clean K-Fold split."""
    files = ["labels_train.csv", "labels_val.csv", "labels_test.csv"]
    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    
    df_full = pd.concat(dfs, ignore_index=True)
    images, labels = [], []

    for _, row in df_full.iterrows():
        img = Image.open(row["filename"]).convert("L").resize((img_size, img_size))
        images.append(np.array(img).flatten() / 255.0)
        labels.append(row["label"])

    return jnp.array(images), 2 * jnp.array(labels) - 1, df_full

# ------------------------- YOUR ORIGINAL FRQI state builder ----------------------

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

# ----------------------- Main K-Fold Logic -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="./kfold_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    X_all, y_all, df_all = load_full_dataset(args.data_dir, args.img_size)
    
    # 2. Setup K-Fold
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"Starting {args.k_folds}-Fold CV with Early Stopping for {args.img_size}x{args.img_size}...")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_all)):
        fold_id = fold + 1
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n--- FOLD {fold_id}/{args.k_folds} ---")
        
        # Split Train into Train and Validation (for Early Stopping)
        X_train_val, X_test = X_all[train_val_idx], X_all[test_idx]
        y_train_val, y_test = y_all[train_val_idx], y_all[test_idx]
        
        # Use 15% of the training fold for validation (Early Stopping)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

        # Initialize Model
        qnode, weight_shape, num_wires = build_qnode(args.img_size, args.num_layers)
        key = jax.random.PRNGKey(fold) 
        params = {
            "q_weights": 0.01 * jax.random.normal(key, weight_shape),
            "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))
        }
        optimizer = optax.adam(args.lr)
        opt_state = optimizer.init(params)

        # Early Stopping Variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_params = params
        loss_history = []

        # Training Loop
        for epoch in range(args.max_epochs):
            batch_losses = []
            for i in range(0, len(X_train), args.batch_size):
                xb, yb = X_train[i : i + args.batch_size], y_train[i : i + args.batch_size]
                params, opt_state, loss = train_step(params, opt_state, xb, yb, qnode, optimizer)
                batch_losses.append(float(loss))
            
            epoch_train_loss = np.mean(batch_losses)
            loss_history.append(epoch_train_loss)

            # Validation Check for Early Stopping
            val_loss = float(soft_margin_loss(params, X_val, y_val, qnode))
            
            if val_loss < (best_val_loss - 1e-4):
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_params = params
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Val Loss={val_loss:.4f}")

            if epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                params = best_params
                break

        # --- EVALUATION FOR THIS FOLD ---
        test_preds = vmap(lambda x: model_forward(params, x, qnode))(X_test)
        test_preds = np.array(test_preds)
        y_test_np = np.array(y_test)
        pred_labels = np.sign(test_preds)
        
        y_bin = (y_test_np + 1) // 2
        pred_bin = (pred_labels + 1) // 2

        # Metrics
        acc = accuracy_score(y_test_np, pred_labels)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        auc = roc_auc_score(y_bin, test_preds)
        
        print(f"  Fold {fold_id} Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        fold_results.append({
            "fold": fold_id, "accuracy": acc, "precision": prec, 
            "recall": rec, "f1": f1, "auc": auc
        })

        # Save Fold-specific plots (Loss, CM, ROC)
        plt.figure(figsize=(7,4)); plt.plot(loss_history); plt.savefig(os.path.join(fold_dir, "loss.png")); plt.close()
        cm = confusion_matrix(y_bin, pred_bin)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Polyp"])
        disp.plot(cmap="Blues"); plt.savefig(os.path.join(fold_dir, "confusion_matrix.png")); plt.close()
        fpr, tpr, _ = roc_curve(y_bin, test_preds)
        plt.figure(figsize=(6,6)); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig(os.path.join(fold_dir, "roc_curve.png")); plt.close()

    # --- FINAL AGGREGATION ---
    df_res = pd.DataFrame(fold_results)
    df_res.to_csv(os.path.join(args.output_dir, "kfold_metrics.csv"), index=False)
    
    summary = {
        "mean_accuracy": float(df_res['accuracy'].mean()),
        "std_accuracy": float(df_res['accuracy'].std()),
        "mean_auc": float(df_res['auc'].mean()),
        "mean_f1": float(df_res['f1'].mean())
    }
    
    with open(os.path.join(args.output_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*40)
    print(f"FINAL K-FOLD RESULTS: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()