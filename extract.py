import re
import matplotlib.pyplot as plt

# Step 1: Read the log file
with open("hybrid.txt", "r") as f:
    raw_log = f.read()

# Step 2: Extract all dictionary-like strings
dict_pattern = r"\{[^{}]+\}"
matches = re.findall(dict_pattern, raw_log)

# Step 3: Parse the strings into actual dictionaries
parsed = []
for entry in matches:
    try:
        parsed.append(eval(entry))
    except:
        continue

# Step 4: Categorize logs
train_logs = []
eval_logs = []
final_log = None

for log in parsed:
    if "eval_loss" in log:
        eval_logs.append(log)
    elif "train_runtime" in log:
        final_log = log
    elif "loss" in log and "mean_token_accuracy" in log:
        train_logs.append(log)

# Step 5: Extract arrays
train_epochs = [log['epoch'] for log in train_logs]
train_losses = [log['loss'] for log in train_logs]
train_accuracies = [log['mean_token_accuracy'] for log in train_logs]

eval_epochs = [log['epoch'] for log in eval_logs]
eval_losses = [log['eval_loss'] for log in eval_logs]
eval_accuracies = [log['eval_mean_token_accuracy'] for log in eval_logs]

# Step 6: Plotting
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_losses, label='Train Loss', marker='o')
plt.plot(eval_epochs, eval_losses, label='Eval Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(eval_epochs, eval_accuracies, label='Eval Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: Final summary
if final_log:
    print("\nFinal Training Summary:")
    for k, v in final_log.items():
        print(f"  {k}: {v}")
