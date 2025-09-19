import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
file_paths = ["file1.csv", "file2.csv", "file3.csv"]  # Replace with your actual paths
ticket_ids = ["T001", "T002", "T003", "T004", "T005"]  # Replace with your actual ticket IDs

# === Load and Filter Data ===
dataframes = []
for path in file_paths:
    df = pd.read_csv(path)
    filtered = df[df["ticket_id"].isin(ticket_ids)].copy()
    dataframes.append(filtered)

# === Create Plot ===
plt.figure(figsize=(12, 6))
markers = ["o", "s", "D"]
colors = ["blue", "green", "red"]
labels = ["File 1", "File 2", "File 3"]

# X-axis will be ticket index (1, 2, 3...) repeated for each file
for i, df in enumerate(dataframes):
    x = list(range(1, len(ticket_ids) + 1))
    y = []
    for tid in ticket_ids:
        row = df[df["ticket_id"] == tid]
        if not row.empty:
            y.append(row.iloc[0]["semantic_resolution_score"])
        else:
            y.append(None)  # If ticket ID missing in file
    plt.plot(x, y, marker=markers[i], color=colors[i], label=labels[i], linestyle='-', linewidth=2)

# === Style Plot ===
plt.xlabel("Ticket Number")
plt.ylabel("Semantic Resolution Score")
plt.title("Semantic Resolution Scores per Ticket Across 3 Files")
plt.xticks(range(1, len(ticket_ids) + 1), labels=[str(i) for i in range(1, len(ticket_ids) + 1)])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
