import ast
import re
import matplotlib.pyplot as plt
LOG_FILE = "hybrid.txt"

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
# === PLOTTING ===
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Train Loss", color="blue", marker="o")
plt.plot(eval_epochs, eval_losses, label="Eval Loss", color="red", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
