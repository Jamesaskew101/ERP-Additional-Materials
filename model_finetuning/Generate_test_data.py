import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

#  Load full dataset 
df = pd.read_csv("final_output.csv")

# Clean 'assistant' prefix in problem & solution, as this appears quite alot 
df["CLEAN_PROBLEM"] = df["CLEAN_PROBLEM"].astype(str).str.removeprefix("assistant").str.strip()
df["RESOLUTION_SUMMARY"] = df["RESOLUTION_SUMMARY"].astype(str).str.removeprefix("assistant").str.strip()

# Create instruction-response pairs, want the model to take in the problem, then output category, urgency code and resolution steps summary
records = []
for row in df.to_dict(orient="records"):
    instruction = (
        "Classify and resolve this support ticket using the following details:\n"
        f"- Problem: {row.get('CLEAN_PROBLEM', '')}\n"
    )

    output = (
        f"- Urgency: {row.get('URGENCYCODE', '')}\n"
        f"- Category: {row.get('CATEGORY', '')}\n"
        f"- Solution: {row.get('RESOLUTION_SUMMARY', '')}\n"
    )

    records.append({
        "instruction": instruction.strip(),
        "output": output.strip()
    })

# Train-test split (95% train, 5% test) 
train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)

#  Save train/test CSVs (keep original columns, cleaned)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Save train/test JSONLs (instruction/output only) for training the model
records_df = pd.DataFrame(records)
train_dataset = Dataset.from_pandas(records_df.iloc[train_df.index])
test_dataset = Dataset.from_pandas(records_df.iloc[test_df.index])

train_dataset.to_json("train.jsonl", orient="records", lines=True)
test_dataset.to_json("test.jsonl", orient="records", lines=True)

print(f"✅ Saved {len(train_df)} train and {len(test_df)} test examples (CSV cleaned + JSONL for training).")
