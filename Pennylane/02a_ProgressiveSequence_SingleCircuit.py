"""
Run as:
    python3 02a_ProgressiveSequence_SingleCircuit.py --data-dir ./Datasets/dataset_250_4by4 --img-size 4 --output-dir ./Results
"""

import os
# Forces JAX to allocate memory as needed
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 

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

# ------------------------- Data loading ----------------------------

def load_image_dataset(labels_file, img_size=4):
    df = pd.read_csv(labels_file)
    images, labels = [], []
    for _, row in df.iterrows():
        img = Image.open(row["filename"]).convert("L").resize((img_size, img_size))
        images.append(np.array(img).flatten() / 255.0)
        labels.append(row["label"])
    
    X = jnp.array(images)
    y = 2 * jnp.array(labels) - 1  # {0,1} → {-1,1}
    return X, y, df

# ------------------------- Circuit Builder -------------------------

def build_dynamic_qnode(img_size, num_layers):
    num_pixels = img_size * img_size
    num_pos_qubits = int(np.log2(num_pixels))
    num_wires = num_pos_qubits + 1
    dev = qml.device("default.qubit", wires=num_wires)

    # 'active_depth' is static so JAX recompiles when a new layer is "turned on"
    @qml.qnode(dev, interface="jax")
    def qnode(inputs, weights, active_depth):
        cos_vals = jnp.cos(jnp.pi * inputs)
        sin_vals = jnp.sin(jnp.pi * inputs)
        state = jnp.stack([cos_vals, sin_vals], axis=-1).reshape(-1)
        state = state / (jnp.linalg.norm(state) + 1e-12)
        qml.StatePrep(state, wires=range(num_wires))

        # We only loop up to active_depth, even if weights has more rows
        for l in range(active_depth):
            for i in range(num_wires):
                qml.RY(weights[l, i, 0], wires=i)
            for i in range(num_wires):
                qml.CRZ(weights[l, i, 1], wires=[i, (i + 1) % num_wires])
                
        return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]
    
    return qnode, num_wires

# ------------------------- Loss & Model ---------------------------

def model_forward(params, x, qnode, active_depth):
    obs_vec = jnp.array(qnode(x, params["q_weights"], active_depth))
    return jnp.dot(params["obs_weights"], obs_vec)

def soft_margin_loss(params, x_batch, y_batch, qnode, active_depth):
    preds = vmap(lambda x: model_forward(params, x, qnode, active_depth))(x_batch)
    return jnp.mean(jnp.logaddexp(0.0, -y_batch * preds))

# active_depth is static to trigger re-jit when a new layer is added
@partial(jit, static_argnames=("qnode", "optimizer_func", "active_depth"))
def train_step(params, opt_state, x_batch, y_batch, qnode, optimizer_func, active_depth):
    loss, grads = value_and_grad(soft_margin_loss)(params, x_batch, y_batch, qnode, active_depth)
    updates, opt_state = optimizer_func.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# --------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--img-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./grid_results")
    parser.add_argument("--epochs-per-layer", type=int, default=5000)
    args = parser.parse_args()

    X_train, y_train, _ = load_image_dataset(os.path.join(args.data_dir, "labels_train.csv"), args.img_size)
    X_test, y_test, df_test = load_image_dataset(os.path.join(args.data_dir, "labels_test.csv"), args.img_size)

    max_layers = 20
    qnode, num_wires = build_dynamic_qnode(args.img_size, max_layers)

    # SETUP CONTINUOUS OPTIMIZER
    batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = max_layers * args.epochs_per_layer * steps_per_epoch
    
    lr_schedule = optax.cosine_decay_schedule(init_value=0.05, decay_steps=total_steps, alpha=0.01)
    optimizer = optax.adam(learning_rate=lr_schedule)
    
    # INITIALIZE PARAMETERS
    key = jax.random.PRNGKey(42)
    initial_layer = 0.01 * jax.random.normal(key, (1, num_wires, 2))
    rest_layers = jnp.zeros((max_layers - 1, num_wires, 2))
    
    params = {
        "q_weights": jnp.concatenate([initial_layer, rest_layers], axis=0),
        "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))
    }
    opt_state = optimizer.init(params)
    
    global_loss_history = []
    summary = []

    # CONTINUOUS PROGRESSIVE LOOP
    for active_depth in range(1, max_layers + 1):
        run_name = f"progressive_depth_{active_depth}"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n>>> TRAINING DEPTH {active_depth} / {max_layers}")
        
        best_train_loss = float('inf')
        best_params_depth = params
        layer_loss_history = []
        train_start_time = time.time()

        for epoch in range(args.epochs_per_layer):
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, len(X_train))
            X_shuff, y_shuff = X_train[idx], y_train[idx]
            
            batch_losses = []
            for i in range(0, len(X_train), batch_size):
                xb, yb = X_shuff[i : i + batch_size], y_shuff[i : i + batch_size]
                params, opt_state, loss = train_step(params, opt_state, xb, yb, qnode, optimizer, active_depth)
                batch_losses.append(float(loss))
            
            avg_loss = np.mean(batch_losses)
            layer_loss_history.append(avg_loss)
            global_loss_history.append(avg_loss)
            
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_params_depth = params

            if (epoch + 1) % 100 == 0:
                print(f"Layer {active_depth} - Epoch {epoch+1}/{args.epochs_per_layer} - Loss: {avg_loss:.6f}")

        total_train_time = time.time() - train_start_time

        # --- EVALUATION AND SAVING (Per Layer) ---
        # Evaluate using the best params found in THIS depth segment
        eval_params = best_params_depth 
        test_preds = np.array(vmap(lambda x: model_forward(eval_params, x, qnode, active_depth))(X_test))
        pred_labels = np.sign(test_preds)
        
        y_test_np = np.array(y_test)
        y_bin = (y_test_np + 1) // 2
        pred_bin = (pred_labels + 1) // 2

        acc = (pred_labels == y_test_np).mean()
        prec = precision_score(y_bin, pred_bin, zero_division=0)
        rec = recall_score(y_bin, pred_bin, zero_division=0)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        try:
            auc = roc_auc_score(y_bin, test_preds)
            fpr, tpr, _ = roc_curve(y_bin, test_preds)
        except:
            auc, fpr, tpr = float("nan"), [0, 1], [0, 1]

        # 1. JSON Metrics
        metrics = {
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1_score": float(f1), "auc": float(auc) if not np.isnan(auc) else None,
            "best_train_loss": float(best_train_loss), "training_time_sec": total_train_time,
            "epochs_completed": args.epochs_per_layer, 
            "config": {"num_layers": active_depth, "lr": 0.05, "batch_size": batch_size, "img_size": args.img_size}
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 2. CSVs
        pd.DataFrame({"epoch": range(1, len(layer_loss_history)+1), "loss": layer_loss_history}).to_csv(os.path.join(run_dir, "loss_history.csv"), index=False)
        pd.DataFrame({"filename": df_test["filename"], "true_label": df_test["label"], "pred_score": test_preds, "pred_label": pred_bin.astype(int)}).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        
        cm = confusion_matrix(y_bin, pred_bin)
        pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]).to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

        # 3. Plots
        plt.figure(figsize=(7,4)); plt.plot(layer_loss_history); plt.title(f"Layer {active_depth} Loss"); plt.savefig(os.path.join(run_dir, "training_loss.png")); plt.close()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot(cmap="Blues"); plt.savefig(os.path.join(run_dir, "confusion_matrix.png")); plt.close()
        plt.figure(figsize=(6,6)); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()

        # 4. Weights
        with open(os.path.join(run_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(eval_params, f)

        summary.append({"num_layers": active_depth, "test_accuracy": float(acc)})
        pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "progressive_summary.csv"), index=False)

    # Final Overall Plot
    plt.figure(figsize=(15, 5))
    plt.plot(global_loss_history)
    for i in range(1, max_layers):
        plt.axvline(x=i * args.epochs_per_layer, color='r', linestyle='--', alpha=0.3)
    plt.title("Full Continuous Progressive Training Loss")
    plt.savefig(os.path.join(args.output_dir, "full_continuous_loss.png"))

if __name__ == "__main__":
    main()