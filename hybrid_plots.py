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

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('rag_test_results_with_similarity.csv')

# Filter out rows where 'model_output' has fewer than 29 words
df['word_count'] = df['model_output'].astype(str).apply(lambda x: len(x.split()))
filtered_df = df[df['word_count'] >= 29]

# Ensure columns exist before proceeding
required_columns = ['similarity_1', 'semantic_resolution_score', 'fuzzy_ratio']
if all(col in filtered_df.columns for col in required_columns):
    # Adjust semantic_resolution_score by adding (fuzzy_ratio / 100)
    filtered_df['adjusted_semantic_score'] = (
        filtered_df['semantic_resolution_score'] + (filtered_df['fuzzy_ratio'] / 100)
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['similarity_1'], filtered_df['adjusted_semantic_score'],
                alpha=0.7, color='green')
    
    plt.xlabel('Similarity 1')
    plt.ylabel('Adjusted Semantic Resolution Score')
    plt.title('Similarity vs Adjusted Semantic Resolution Score (Filtered)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Calculate correlation statistics
    correlation = filtered_df[['similarity_1', 'adjusted_semantic_score']].corr(method='pearson')
    print("Correlation matrix (Pearson):")
    print(correlation)
    
    # Optionally, compute and print Spearman and Kendall correlations as well
    spearman_corr = filtered_df[['similarity_1', 'adjusted_semantic_score']].corr(method='spearman')
    kendall_corr = filtered_df[['similarity_1', 'adjusted_semantic_score']].corr(method='kendall')
    
    print("\nSpearman correlation:")
    print(spearman_corr)
    
    print("\nKendall correlation:")
    print(kendall_corr)


