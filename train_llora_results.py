import ast
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

LOG_FILE = "llora_results.txt"

train_accs, train_losses, train_epochs = [], [], []
eval_accs, eval_losses, eval_entropies, eval_epochs = [], [], [], []

# Match a dictionary object in the line using regex
pattern = re.compile(r"\{.*\}")

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        try:
            metrics = ast.literal_eval(match.group())

            # Evaluation metrics
            if "eval_mean_token_accuracy" in metrics or "eval_loss" in metrics:
                if "eval_mean_token_accuracy" in metrics:
                    eval_accs.append(metrics["eval_mean_token_accuracy"])
                if "eval_loss" in metrics:
                    eval_losses.append(metrics["eval_loss"])
                if "eval_entropy" in metrics:
                    eval_entropies.append(metrics["eval_entropy"])
                if "epoch" in metrics:
                    eval_epochs.append(metrics["epoch"])

            # Training metrics
            elif "mean_token_accuracy" in metrics and "loss" in metrics:
                train_accs.append(metrics["mean_token_accuracy"])
                train_losses.append(metrics["loss"])
                if "epoch" in metrics:
                    train_epochs.append(metrics["epoch"])

        except Exception as e:
            print(f"Error parsing line: {e}")
            continue

def avg(lst):
    return round(sum(lst) / len(lst), 5) if lst else None

print("=== TRAINING METRICS ===")
print(f"Avg Train Accuracy: {avg(train_accs)}")
print(f"Avg Train Loss:     {avg(train_losses)}")
print(f"Epoch Range:        {min(train_epochs)} – {max(train_epochs)}" if train_epochs else "No training data")

print("\n=== EVALUATION METRICS ===")
print(f"Avg Eval Accuracy:  {avg(eval_accs)}")
print(f"Avg Eval Loss:      {avg(eval_losses)}")
print(f"Avg Eval Entropy:   {avg(eval_entropies)}")
print(f"Epoch Range:        {min(eval_epochs)} – {max(eval_epochs)}" if eval_epochs else "No evaluation data")

# Debug info
print(f"\n[DEBUG] Eval entries found: {len(eval_accs)}")
print(f"\n[DEBUG] Eval entries found: {len(eval_accs)}")

# Convert lists to numpy arrays for smoothing
train_epochs_np = np.array(train_epochs)
train_losses_np = np.array(train_losses)
eval_epochs_np = np.array(eval_epochs)
eval_losses_np = np.array(eval_losses)

# === SMOOTHING ===
# Use only if enough data points exist
# === SAFE SMOOTH FUNCTION ===
def safe_smooth(y, window=9, poly=2):
    return savgol_filter(
        y,
        window_length=min(len(y) if len(y) % 2 == 1 else len(y) - 1, window),
        polyorder=poly
    ) if len(y) > 5 else y

# Smooth the curves
train_losses_smooth = safe_smooth(train_losses_np)
eval_losses_smooth = safe_smooth(eval_losses_np)

# === PLOTTING ===
plt.figure(figsize=(10, 6))

# Smoothed Training Loss
plt.plot(
    train_epochs_np[:len(train_losses_smooth)],
    train_losses_smooth,
    label="Training Loss",
    color="blue",
    linewidth=2.5
)

# Smoothed Validation Loss
plt.plot(
    eval_epochs_np[:len(eval_losses_smooth)],
    eval_losses_smooth,
    label="Validation Loss",
    color="red",
    linewidth=2.5
)

# === Formatting ===
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training and Validation Loss Across Epochs", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0.6, 3.8)  # tighten range to your data
plt.tight_layout()
plt.show()