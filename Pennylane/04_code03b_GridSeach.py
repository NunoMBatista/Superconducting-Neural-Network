"""
Run as:
    python3 04_code03b_GridSeach.py --data-dir ./Datasets/dataset_250_4by4 --img-size 4 --max-epochs 20 --patience 10 --output-dir ./Code02_epochs20
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" #forces JAX to not reserve about 90% of VRAM at the beginning
import warnings
import time
import json
import argparse
import itertools
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
import pickle
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore", message="PennyLane is currently not compatible with versions of JAX")

# --- Helper Functions (No changes here) ---
def load_image_dataset(labels_file, img_size=4, sample_size=None):
    df = pd.read_csv(labels_file)
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    images, labels = [], []
    for _, row in df.iterrows():
        img = Image.open(row["filename"]).convert("L").resize((img_size, img_size))
        images.append(np.array(img).flatten() / 255.0)
        labels.append(row["label"])
    return jnp.array(images), 2 * jnp.array(labels) - 1, df

def build_qnode(img_size, num_layers):
    num_pixels = img_size * img_size
    num_pos_qubits = int(np.log2(num_pixels))
    num_wires = num_pos_qubits + 1
    dev = qml.device("default.qubit", wires=num_wires)
    @qml.qnode(dev, interface="jax")
    def qnode(inputs, weights):
        flat = inputs.flatten()
        cos_vals, sin_vals = jnp.cos(jnp.pi * flat), jnp.sin(jnp.pi * flat)
        state = jnp.stack([cos_vals, sin_vals], axis=-1).reshape(-1)
        state = state / (jnp.linalg.norm(state) + 1e-12)
        qml.StatePrep(state, wires=range(num_wires))
        qml.BasicEntanglerLayers(weights, wires=range(num_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]
    return qnode, (num_layers, num_wires), num_wires

def model_forward(params, x, qnode):
    obs_vec = jnp.array(qnode(x, params["q_weights"]))
    return jnp.dot(params["obs_weights"], obs_vec)

def soft_margin_loss(params, x_batch, y_batch, qnode):
    preds = vmap(lambda x: model_forward(params, x, qnode))(x_batch)
    return jnp.mean(jnp.logaddexp(0.0, -y_batch * preds))

@partial(jit, static_argnames=("qnode", "optimizer"))
def train_step(params, opt_state, x_batch, y_batch, qnode, optimizer):
    loss, grads = value_and_grad(soft_margin_loss)(params, x_batch, y_batch, qnode)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# --- The Updated Runner with Early Stopping ---
def run_and_save(config, data, run_dir, max_epochs, patience=5, min_delta=1e-4):
    X_train, y_train, X_val, y_val, X_test, y_test, df_test = data
    os.makedirs(run_dir, exist_ok=True)
    
    qnode, weight_shape, num_wires = build_qnode(config['img_size'], config['num_layers'])
    optimizer = optax.adam(config['lr']) if config['opt_name'] == 'adam' else optax.sgd(config['lr'])
    
    key = jax.random.PRNGKey(42)
    params = {"q_weights": 0.01 * jax.random.normal(key, weight_shape),
              "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))}
    opt_state = optimizer.init(params)

    loss_history, metrics_history = [], []
    train_start = time.time()
    
    # Early Stopping Variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    final_epoch_count = 0

    for epoch in range(max_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, len(X_train))
        X_shuff, y_shuff = X_train[idx], y_train[idx]
        
        batch_losses = []
        for i in range(0, len(X_train), config['batch_size']):
            xb, yb = X_shuff[i : i + config['batch_size']], y_shuff[i : i + config['batch_size']]
            params, opt_state, loss = train_step(params, opt_state, xb, yb, qnode, optimizer)
            batch_losses.append(float(loss))
        
        epoch_train_loss = np.mean(batch_losses)
        
        # Calculate Validation Loss for Early Stopping
        val_loss = float(soft_margin_loss(params, X_val, y_val, qnode))
        preds_v = vmap(lambda x: model_forward(params, x, qnode))(X_val)
        val_acc = (np.sign(np.array(preds_v)) == np.array(y_val)).mean()
        
        loss_history.append(epoch_train_loss)
        metrics_history.append({"epoch": epoch + 1, "loss": epoch_train_loss, "val_loss": val_loss, "accuracy": float(val_acc)})
        final_epoch_count = epoch + 1

        print(f"  Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early Stopping Logic
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the "best" parameters encountered so far
            best_params = params
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            params = best_params # Revert to best params for final eval
            break

    train_time = time.time() - train_start

    # --- SAVE ALL OUTPUT FILES (Same as your request) ---
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

    pd.DataFrame({"epoch": range(1, len(loss_history)+1), "loss": loss_history}).to_csv(os.path.join(run_dir, "loss_history.csv"), index=False)
    pd.DataFrame(metrics_history).to_csv(os.path.join(run_dir, "metrics_by_epoch.csv"), index=False)
    
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1_score": float(f1), "auc": float(auc) if not np.isnan(auc) else None,
        "training_time_sec": float(train_time), "epochs_completed": final_epoch_count, "config": config
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"filename": df_test["filename"], "true_label": df_test["label"], "pred_score": test_preds, "pred_label": pred_bin.astype(int)}).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    
    plt.figure(figsize=(7,4)); plt.plot(loss_history); plt.title("Training Loss"); plt.savefig(os.path.join(run_dir, "training_loss.png")); plt.close()
    
    cm = confusion_matrix(y_bin, pred_bin)
    pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]).to_csv(os.path.join(run_dir, "confusion_matrix.csv"))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues"); plt.savefig(os.path.join(run_dir, "confusion_matrix.png")); plt.close()

    plt.figure(figsize=(6,6)); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()

    return acc, final_epoch_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--img-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50) # Increased default
    parser.add_argument("--patience", type=int, default=5) 
    parser.add_argument("--output-dir", type=str, default="./grid_results")
    args = parser.parse_args()

    grid = {
        "num_layers": [1, 2, 3, 5, 7, 10, 12, 15, 20, 30],
        "lr": [0.001, 0.01, 0.02, 0.05],
        "batch_size": [8, 16, 32, 64],
        "opt_name": ["adam", "sgd"],
    }
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    X_train, y_train, _ = load_image_dataset(os.path.join(args.data_dir, "labels_train.csv"), args.img_size)
    X_val, y_val, _ = load_image_dataset(os.path.join(args.data_dir, "labels_val.csv"), args.img_size)
    X_test, y_test, df_test = load_image_dataset(os.path.join(args.data_dir, "labels_test.csv"), args.img_size)
    data = (X_train, y_train, X_val, y_val, X_test, y_test, df_test)

    summary_results = []
    for i, config in enumerate(combinations):
        config['img_size'] = args.img_size
        folder_name = f"run_{i:02d}_{config['opt_name']}_ly{config['num_layers']}_lr{config['lr']}_bs{config['batch_size']}"
        run_dir = os.path.join(args.output_dir, folder_name)
        
        print(f"\n>>> Running Config {i+1}/{len(combinations)}: {folder_name}")
        test_acc, stopped_at = run_and_save(config, data, run_dir, args.max_epochs, args.patience)
        
        summary_results.append({**config, "test_accuracy": test_acc, "epochs_run": stopped_at, "folder": folder_name})
        pd.DataFrame(summary_results).to_csv(os.path.join(args.output_dir, "grid_search_summary.csv"), index=False)

if __name__ == "__main__":
    main()