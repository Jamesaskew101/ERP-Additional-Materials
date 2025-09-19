#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced plot: Raw vs Hybrid similarity with similarity_1 overlay.
Clear visual separation between high and low tickets, each ordered by similarity_1.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load both CSVs
df_raw = pd.read_csv("raw_results.csv")
df_hybrid = pd.read_csv("hybrid_results.csv")

#Randomly selected TickID's from top and bottom 50 similar cases
high_ids = [
    "t6UJ9A00FADU", "t6UJ9A00FN95", "t6UJ9A00F1PR", "t6UJ9A00FQAT", "t6UJ9A00FH43",
    "t6UJ9A00G4K5", "t6UJ9A00FSKT", "t6UJ9A00FCZP", "t6UJ9A00FUN1", "t6UJ9A00G06Y"
]
low_ids = [
    "t6UJ9A00EZ3Z", "t6UJ9A00F6OC", "t6UJ9A00FPPH", "t6UJ9A00FPPH", "t6UJ9A00GEPX",
    "t6UJ9A00FQGZ", "t6UJ9A00G3E0"
]

# Extract similarities and similarity_1
def get_sim_data(df, ticket_ids):
    df_filtered = df[df['ticketID'].astype(str).isin(ticket_ids)]
    df_filtered = df_filtered.set_index('ticketID')
    sim = df_filtered['resolution_embedding_similarity'].to_dict()
    sim1 = df_filtered['similarity_1'].to_dict() if 'similarity_1' in df_filtered.columns else {tid: None for tid in ticket_ids}
    return sim, sim1

raw_sims, raw_sim1 = get_sim_data(df_raw, high_ids + low_ids)
hybrid_sims, hybrid_sim1 = get_sim_data(df_hybrid, high_ids + low_ids)

# Sort high and low ids separately based on raw_sim1
high_sorted = sorted(high_ids, key=lambda tid: raw_sim1.get(tid, 0), reverse=True)
low_sorted = sorted(low_ids, key=lambda tid: raw_sim1.get(tid, 0), reverse=True)
selected_ids = high_sorted + low_sorted

# Prepare values in new order
raw_values = [raw_sims.get(tid, 0) for tid in selected_ids]
hybrid_values = [hybrid_sims.get(tid, 0) for tid in selected_ids]
raw_sim1_values = [raw_sim1.get(tid, None) for tid in selected_ids]
hybrid_sim1_values = [hybrid_sim1.get(tid, None) for tid in selected_ids]

# Plotting
x = range(len(selected_ids))
bar_width = 0.35

plt.figure(figsize=(16, 6))
plt.bar(x, raw_values, width=bar_width, label="Raw", color='#4B8BBE', alpha=0.8)
plt.bar([i + bar_width for i in x], hybrid_values, width=bar_width, label="Hybrid", color='#FFBD44', alpha=0.8)

# Add similarity_1 markers
plt.plot([i + bar_width/2 for i in x], raw_sim1_values, 'ko', label="Raw similarity_1")


# Labels and layout
plt.xticks([i + bar_width / 2 for i in x], selected_ids, rotation=45, ha='right')
plt.ylabel("Resolution Embedding Similarity", fontsize=12)
plt.title("Embedding Similarity by Ticket (Raw vs Hybrid), Grouped by High/Low, Ordered by Raw similarity_1\n", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.axvline(x=len(high_sorted) - 0.5, color='gray', linestyle='--', linewidth=1)

# Add High/Low labels
plt.text(len(high_sorted)/2 - 0.5, plt.ylim()[1]*1.01, 'High Similarity Tickets',
         ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.text(len(high_sorted) + len(low_sorted)/2 - 0.5, plt.ylim()[1]*1.01, 'Low Similarity Tickets',
         ha='center', va='bottom', fontsize=11, fontweight='bold')

# Final touches
plt.legend()
plt.tight_layout()
plt.show()

# Step 1: Load CSVs (already done)
# df_raw = pd.read_csv("raw_results.csv")
# df_hybrid = pd.read_csv("hybrid_results.csv")
# Step 1: Merge Raw and Hybrid on ticketID and keep similarity_1 from Raw
df_raw['ticketID'] = df_raw['ticketID'].astype(str)
df_hybrid['ticketID'] = df_hybrid['ticketID'].astype(str)

df_compare = pd.merge(
    df_raw[['ticketID', 'resolution_embedding_similarity', 'similarity_1']],
    df_hybrid[['ticketID', 'resolution_embedding_similarity']],
    on='ticketID',
    suffixes=('_raw', '_hybrid')
)

# Step 2: Drop rows where similarity_1 is missing and sort by similarity_1
df_sorted = df_compare.dropna(subset=['similarity_1']).sort_values(by='similarity_1', ascending=False)

# Step 3: Top 50 and Bottom 50 based on similarity_1
top_50 = df_sorted.head(50)
bottom_50 = df_sorted.tail(50)

# Step 4: Calculate performance comparisons
def compute_better_percentage(df_slice):
    better = (df_slice['resolution_embedding_similarity_hybrid'] > df_slice['resolution_embedding_similarity_raw']).sum()
    total = len(df_slice)
    return better / total * 100

top_percent = compute_better_percentage(top_50)
bottom_percent = compute_better_percentage(bottom_50)

# Output results
print(f"Hybrid outperforms Raw in:")
print(f"🔼 Top 50 (by similarity_1): {top_percent:.1f}% of cases")
print(f"🔽 Bottom 50 (by similarity_1): {bottom_percent:.1f}% of cases")


import matplotlib.pyplot as plt
import numpy as np

# Your metrics
hybrid_metrics = {
    "Urgency Accuracy": 88.37,
    "Category Accuracy": 74.03,
    "Fuzzy Score": 51.77,
    "Embedding Similarity": 0.4693 * 100  # convert to percentage for consistency
}

norag_metrics = {
    "Urgency Accuracy": 89.56,
    "Category Accuracy": 73.64,
    "Fuzzy Score": 44.04,
    "Embedding Similarity": 0.4663 * 100
}

labels = list(hybrid_metrics.keys())
hybrid_vals = [ hybrid_metrics[m] for m in labels ]
norag_vals = [ norag_metrics[m] for m in labels ]

# Number of metrics
num_metrics = len(labels)

# Setup angle for radar
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
# Close the loop
angles += angles[:1]
hybrid_vals += hybrid_vals[:1]
norag_vals += norag_vals[:1]

# Plot
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], labels, color='black', size=10)

# Draw the two polygons
ax.plot(angles, hybrid_vals, color='blue', linewidth=2, label='Hybrid Model')
ax.fill(angles, hybrid_vals, color='blue', alpha=0.25)

ax.plot(angles, norag_vals, color='red', linewidth=2, label='Raw Llama-3')
ax.fill(angles, norag_vals, color='red', alpha=0.25)

# Optional: set radial grid and range
ax.set_rlabel_position(30)
max_val = max(max(hybrid_vals), max(norag_vals))
ax.set_ylim(0, max_val + 10)

# Title and legend
plt.title("Comparison of Hybrid vs No‑Retriever Metrics", size=14, y=1.05)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Load hybrid data
df_hybrid = pd.read_csv("hybrid_results.csv")

# Clean NaNs
df_kde = df_hybrid.dropna(subset=["similarity_1", "fuzzy_ratio"])

# Compute Pearson correlation
r, p = pearsonr(df_kde["similarity_1"], df_kde["fuzzy_ratio"])
print(f"📈 Pearson Correlation (similarity_1 vs fuzzy_ratio): {r:.4f}")
print(f"📊 P-value: {p:.4e}")

# Plot
plt.figure(figsize=(8, 6))

# KDE background
sns.kdeplot(
    data=df_kde,
    x="similarity_1",
    y="fuzzy_ratio",
    fill=True,
    cmap="plasma",
    thresh=0.05,
    levels=100,
    alpha=0.8
)

# Regression line (with scatter optionally)
sns.regplot(
    data=df_kde,
    x="similarity_1",
    y="fuzzy_ratio",
    scatter=False,
    color="black",
    line_kws={"label": f"Pearson r = {r:.2f}"}
)

# Final plot details
plt.title("Cosine Coeffcient of Retrived Cases against Fuzzy Match Ration", fontsize=14)
plt.xlabel("Cosine Coeffcient of Most Similar Case")
plt.ylabel("Fuzzy Match Ratio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



