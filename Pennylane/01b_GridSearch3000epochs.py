"""
Run as:
    python3 01_GridSearch.py --data-dir ./Datasets/dataset_250_4by4 --img-size 4 --output-dir ./Results
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

def build_qnode(img_size, num_layers):
    num_pixels = img_size * img_size
    num_pos_qubits = int(np.log2(num_pixels))
    num_wires = num_pos_qubits + 1
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev, interface="jax")
    def qnode(inputs, weights):
        cos_vals = jnp.cos(jnp.pi * inputs)
        sin_vals = jnp.sin(jnp.pi * inputs)
        state = jnp.stack([cos_vals, sin_vals], axis=-1).reshape(-1)
        state = state / (jnp.linalg.norm(state) + 1e-12)
        
        qml.StatePrep(state, wires=range(num_wires))
        qml.BasicEntanglerLayers(weights, wires=range(num_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]
    
    return qnode, (num_layers, num_wires), num_wires

# ------------------------- Loss & Model ---------------------------

def model_forward(params, x, qnode):
    obs_vec = jnp.array(qnode(x, params["q_weights"]))
    return jnp.dot(params["obs_weights"], obs_vec)

def soft_margin_loss(params, x_batch, y_batch, qnode):
    preds = vmap(lambda x: model_forward(params, x, qnode))(x_batch)
    return jnp.mean(jnp.logaddexp(0.0, -y_batch * preds))

@partial(jit, static_argnames=("qnode", "optimizer_func"))
def train_step(params, opt_state, x_batch, y_batch, qnode, optimizer_func):
    loss, grads = value_and_grad(soft_margin_loss)(params, x_batch, y_batch, qnode)
    updates, opt_state = optimizer_func.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ------------------------- Runner ---------------------------------

def run_experiment(config, data, run_dir, max_epochs=3000):
    X_train, y_train, X_test, y_test, df_test = data
    os.makedirs(run_dir, exist_ok=True)

    jax.clear_caches() 

    qnode, weight_shape, num_wires = build_qnode(config['img_size'], config['num_layers'])
    
    steps_per_epoch = int(np.ceil(len(X_train) / config['batch_size']))
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config['lr'], 
        decay_steps=max_epochs * steps_per_epoch, 
        alpha=0.01
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    
    key = jax.random.PRNGKey(42)
    params = {
        "q_weights": 0.01 * jax.random.normal(key, weight_shape),
        "obs_weights": 0.1 * jax.random.normal(key, (num_wires,))
    }
    opt_state = optimizer.init(params)

    best_train_loss = float('inf')
    best_params = params
    loss_history = []
    train_start_time = time.time()

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
        loss_history.append(epoch_train_loss)

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            best_params = params

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/3000 - Loss: {epoch_train_loss:.6f}")

    total_train_time = time.time() - train_start_time
    
    # Final Evaluation
    params = best_params
    test_preds = vmap(lambda x: model_forward(params, x, qnode))(X_test)
    test_preds = np.array(test_preds)
    pred_labels = np.sign(test_preds)
    
    y_test_np = np.array(y_test)
    y_bin = (y_test_np + 1) // 2
    pred_bin = (pred_labels + 1) // 2

    # Calculations
    acc = (pred_labels == y_test_np).mean()
    prec = precision_score(y_bin, pred_bin, zero_division=0)
    rec = recall_score(y_bin, pred_bin, zero_division=0)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_bin, test_preds)
        fpr, tpr, _ = roc_curve(y_bin, test_preds)
    except:
        auc, fpr, tpr = float("nan"), [0, 1], [0, 1]

    # --- SAVING SECTION ---

    # 1. JSON Metrics
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1_score": float(f1), "auc": float(auc) if not np.isnan(auc) else None,
        "best_train_loss": float(best_train_loss), "training_time_sec": total_train_time,
        "epochs_completed": max_epochs, "config": config
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 2. CSVs (Loss, Predictions, and Confusion Matrix)
    pd.DataFrame({"epoch": range(1, len(loss_history)+1), "loss": loss_history}).to_csv(os.path.join(run_dir, "loss_history.csv"), index=False)
    
    pd.DataFrame({
        "filename": df_test["filename"], "true_label": df_test["label"], 
        "pred_score": test_preds, "pred_label": pred_bin.astype(int)
    }).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    cm = confusion_matrix(y_bin, pred_bin)
    pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]).to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

    # 3. Plots (Loss, CM heatmap, ROC)
    plt.figure(figsize=(7,4)); plt.plot(loss_history); plt.title("Training Loss"); plt.savefig(os.path.join(run_dir, "training_loss.png")); plt.close()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues"); plt.savefig(os.path.join(run_dir, "confusion_matrix.png")); plt.close()

    plt.figure(figsize=(6,6)); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()

    # 4. Weights
    with open(os.path.join(run_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(params, f)
    
    return float(acc)

# --------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--img-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./grid_results")
    args = parser.parse_args()

    grid = {
        "num_layers": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "lr": [0.05],
        "batch_size": [32],
        "opt_name": ["adam"],
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    X_train, y_train, _ = load_image_dataset(os.path.join(args.data_dir, "labels_train.csv"), args.img_size)
    X_test, y_test, df_test = load_image_dataset(os.path.join(args.data_dir, "labels_test.csv"), args.img_size)
    data = (X_train, y_train, X_test, y_test, df_test)

    summary = []
    for i, config in enumerate(combinations):
        config['img_size'] = args.img_size
        run_name = f"run_{i}_ly{config['num_layers']}_lr{config['lr']}_bs{config['batch_size']}"
        run_dir = os.path.join(args.output_dir, run_name)
        print(f"\n>>> Config {i+1}/{len(combinations)}: {run_name}")
        
        acc = run_experiment(config, data, run_dir, max_epochs=3000)
        
        summary.append({**config, "test_accuracy": acc, "epochs_run": 3000})
        pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "grid_summary.csv"), index=False)

if __name__ == "__main__":
    main()