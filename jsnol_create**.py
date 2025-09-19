import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

#Load the csv file
df = pd.read_csv("final_main_data.csv")

#filter out missing solution or problem text
df = df[df["PROBLEM"].notna() & df["SOLUTION"].notna()]
df = df[(df["PROBLEM"].str.strip() != "") & (df["SOLUTION"].str.strip() != "")]

#split the data into train and test, saving both as csv files
train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
train_df.to_csv("train_95.csv", index=False)
test_df.to_csv("test_5.csv", index=False)

#Now create instruction-output pairs for data training
records = []
for row in train_df.to_dict(orient="records"):
    instruction = (
        "Classify and resolve this support ticket using the following details:\n"
        f"- ticketID: {row.get('TICKETID', '')}\n"
        f"- Subject: {row.get('SUBJECT', '')}\n"
        f"- Problem: {row.get('PROBLEM', '')}\n"
    )
    output = (
        f"- Urgency: {row.get('URGENCYCODE', '')}\n"
        f"- Issue: {row.get('ISSUE', '')}\n"
        f"- Category: {row.get('CATEGORY', '')}\n"
        f"- Solution: {row.get('SOLUTION', '')}\n"
        f"- Flow: {row.get('RESOLUTION_FLOW', '')}\n"
    )
    records.append({
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip()
    })
#save data as a jsnol file so each line is a valid jsonl object
dataset = Dataset.from_list(records)
dataset.to_json("final_train.jsonl", orient="records", lines=True)
problem_csv = train_df[["TICKETID", "PROBLEM"]]
problem_csv.to_csv("train_problems.csv", index=False)
