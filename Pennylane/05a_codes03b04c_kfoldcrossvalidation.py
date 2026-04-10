"""
K-Fold Cross-Validation with Early Stopping for QML.
Backend: default.qubit accelerated via JAX.
Features: FRQI Encoding, Cosine Learning Rate Decay, and Memory Clearing.

Run as:
    python3 05a_codes03b04c_kfoldcrossvalidation.py --data-dir ./Datasets/dataset_250_16by16 --img-size 16 --k-folds 5 --max-epochs 500 --patience 20 --batch-size 16 --lr 0.02 --num-layers 7 --output-dir ./Code02_epochs20
"""

import os
# Force JAX to not reserve 90% of VRAM immediately
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

# ------------------------- Data loading ----------------------------

def load_full_dataset(data_dir, img_size):
    """Combines train, val, and test labels to perform a clean K-Fold split from the whole pool."""
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

# ------------------------- FRQI State Builder ----------------------

def build_frqi_statevector(image):
    """Encodes image data into a quantum state using FRQI."""
    flat = image.flatten()
    cos_vals = jnp.cos(jnp.pi * flat)
    sin_vals = jnp.sin(jnp.pi * flat)
    state = jnp.stack([cos_vals, sin_vals], axis=-1).reshape(-1)
    return state / (jnp.linalg.norm(state) + 1e-12)

# ------------------------- Build QNode ------------------------------

def build_qnode(img_size, num_layers):
    """Initializes the PennyLane device and QNode."""
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

# ----------------------- Hybrid Model Logic -----------------------

def model_forward(params, x, qnode):
    """Hybrid forward pass: Quantum expectation values followed by a linear observer weight."""
    obs_vec = jnp.array(qnode(x, params["q_weights"]))
    return jnp.dot(params["obs_weights"], obs_vec)

def soft_margin_loss(params, x_batch, y_batch, qnode):
    """Soft margin loss (equivalent to Logistic Loss) for binary classification."""
    preds = vmap(lambda x: model_forward(params, x, qnode))(x_batch)
    return jnp.mean(jnp.logaddexp(0.0, -y_batch * preds))

@partial(jit, static_argnames=("qnode", "optimizer_func"))
def train_step(params, opt_state, x_batch, y_batch, qnode, optimizer_func):
    """JIT-compiled training step."""
    loss, grads = value_and_grad(soft_margin_loss)(params, x_batch, y_batch, qnode)
    updates, opt_state = optimizer_func.update(grads, opt_state, params)
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
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./kfold_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data (Concatenating all splits for K-Fold)
    X_all, y_all, _ = load_full_dataset(args.data_dir, args.img_size)
    
    # 2. Setup K-Fold
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"Starting {args.k_folds}-Fold CV for {args.img_size}x{args.img_size} images...")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_all)):
        # Clear JAX cache to prevent memory leaks and compilation clutter
        jax.clear_caches()
        
        fold_id = fold + 1
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n--- FOLD {fold_id}/{args.k_folds} ---")
        
        # Split current fold into training set and test set
        X_train_val, X_test = X_all[train_val_idx], X_all[test_idx]
        y_train_val, y_test = y_all[train_val_idx], y_all[test_idx]
        
        # Internal split for Early Stopping (15% of the training fold)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=42
        )

        # 3. Initialize Model and Optimizer with Cosine Decay
        qnode, weight_shape, num_wires = build_qnode(args.img_size, args.num_layers)
        
        steps_per_epoch = int(np.ceil(len(X_train) / args.batch_size))
        lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr, 
            decay_steps=args.max_epochs * steps_per_epoch, 
            alpha=0.01
        )
        optimizer = optax.adam(learning_rate=lr_schedule)
        
        key = jax.random.PRNGKey(fold_id) 
        params = {
            "q_weights": 0.01 * jax.random.normal(key, weight_shape),
            "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))
        }
        opt_state = optimizer.init(params)

        # Early Stopping Variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_params = params
        loss_history = []

        # 4. Training Loop
        for epoch in range(args.max_epochs):
            # Shuffle training data for this epoch
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, len(X_train))
            X_tr_shuff, y_tr_shuff = X_train[idx], y_train[idx]

            batch_losses = []
            for i in range(0, len(X_train), args.batch_size):
                xb, yb = X_tr_shuff[i : i + args.batch_size], y_tr_shuff[i : i + args.batch_size]
                params, opt_state, loss = train_step(params, opt_state, xb, yb, qnode, optimizer)
                batch_losses.append(float(loss))
            
            epoch_train_loss = np.mean(batch_losses)
            loss_history.append(epoch_train_loss)

            # Validation Check
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
                break
        
        # Load the best parameters from early stopping
        params = best_params

        # 5. Evaluation for this Fold
        test_preds = vmap(lambda x: model_forward(params, x, qnode))(X_test)
        test_preds = np.array(test_preds)
        y_test_np = np.array(y_test)
        pred_labels = np.sign(test_preds)
        
        y_bin = (y_test_np + 1) // 2
        pred_bin = (pred_labels + 1) // 2

        # Metrics calculation
        acc = accuracy_score(y_test_np, pred_labels)
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        try:
            auc = roc_auc_score(y_bin, test_preds)
        except:
            auc = float('nan')
        
        print(f"  Fold {fold_id} Metrics -> Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        fold_results.append({
            "fold": fold_id, "accuracy": acc, "precision": prec, 
            "recall": rec, "f1": f1, "auc": auc, "epochs_completed": epoch + 1
        })

        # 6. Saving Fold-specific plots
        plt.figure(figsize=(7,4))
        plt.plot(loss_history)
        plt.title(f"Fold {fold_id} Loss History")
        plt.savefig(os.path.join(fold_dir, "loss.png"))
        plt.close()

        cm = confusion_matrix(y_bin, pred_bin)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Polyp"])
        disp.plot(cmap="Blues")
        plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
        plt.close()

        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_bin, test_preds)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
            plt.savefig(os.path.join(fold_dir, "roc_curve.png"))
            plt.close()

    # --- Final Aggregation and Summary ---
    df_res = pd.DataFrame(fold_results)
    df_res.to_csv(os.path.join(args.output_dir, "kfold_metrics_detailed.csv"), index=False)
    
    summary = {
        "mean_accuracy": float(df_res['accuracy'].mean()),
        "std_accuracy": float(df_res['accuracy'].std()),
        "mean_auc": float(df_res['auc'].dropna().mean()),
        "mean_f1": float(df_res['f1'].mean()),
        "config": {
            "img_size": args.img_size,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "batch_size": args.batch_size
        }
    }
    
    with open(os.path.join(args.output_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*50)
    print(f"FINAL K-FOLD RESULTS ({args.k_folds} folds)")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean AUC:      {summary['mean_auc']:.4f}")
    print(f"Mean F1-Score: {summary['mean_f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()