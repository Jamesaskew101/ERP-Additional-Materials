import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re
import torch

# Configure load, fine tuned model , embedded model to calculate retiver similarity, and domain specific words. 
MODEL_PATH = "../final_model"
TEST_FILE = "../test.jsonl"
RETRIEVER_DATA_DIR = "../retriever_data"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOMAIN_KEYWORDS_FILE = "../domain_keywords.txt"

#Load Domain specific keywords
with open(DOMAIN_KEYWORDS_FILE, "r") as f:
    domain_keywords = set(line.strip().upper() for line in f if line.strip())

# Load Model & Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Load retriever embeddings and metadata 
embeddings = np.load(f"{RETRIEVER_DATA_DIR}/embeddings.npy")
metadata = pd.read_parquet(f"{RETRIEVER_DATA_DIR}/metadata.parquet")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Load test data 
dataset = load_dataset("json", data_files=TEST_FILE, split="train")

# Create prompt template which includes information from the three most similar cases
instruction_template = """You are a support assistant. Use the urgency code and category from past similar cases to help assess new queries.

### Similar Past Cases:
1. Problem: {p1}
   Resolution: {s1}
   Urgency: {u1}, Category: {c1}

2. Problem: {p2}
   Resolution: {s2}
   Urgency: {u2}, Category: {c2}

3. Problem: {p3}
   Resolution: {s3}
   Urgency: {u3}, Category: {c3}

### User Query:
{query}

### Task:
Using the retrieved examples and your own understanding, predict the **Urgency Code**, **Category**, and **Resolution** for the user's query.

Return your answer strictly in the following JSON format:

{{
  "Urgency Code": "<predicted urgency code>",
  "Category": "<predicted category>",
  "Resolution": "<concise helpful solution>"
}}
"""

def make_inference_prompt(query, similar_cases):
    return f"""### Instruction:
{instruction_template.format(
        p1=similar_cases[0][0], s1=similar_cases[0][1], u1=similar_cases[0][2], c1=similar_cases[0][3],
        p2=similar_cases[1][0], s2=similar_cases[1][1], u2=similar_cases[1][2], c2=similar_cases[1][3],
        p3=similar_cases[2][0], s3=similar_cases[2][1], u3=similar_cases[2][2], c3=similar_cases[2][3],
        query=query
    )}
### Response:
"""

# Inference with retrieval process
results = []

for item in dataset:
    query = item["instruction"]

    # Embed and compute cosine similarity
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    # Compute domain-specific keyword overlap
    domain_boosts = metadata["tokens"].apply(lambda t: len(domain_keywords.intersection(t))).to_numpy()
    domain_boosts_norm = (domain_boosts - domain_boosts.min()) / (domain_boosts.max() - domain_boosts.min() + 1e-8)

    # Hybrid score (cosine + keyword match)
    alpha = 0.95  # cosine similarity weight
    beta = 0.05  # domain keyword match weight
    hybrid_scores = alpha * sims + beta * domain_boosts_norm
    top_idx = hybrid_scores.argsort()[::-1][:3]

    # Select top-3 similar cases
    similar_cases = []
    for i in top_idx:
        row = metadata.iloc[i]
        similar_cases.append((
            row["CLEAN_PROBLEM"],
            row["RESOLUTION_SUMMARY"],
            row["URGENCYCODE"],
            row["CATEGORY"],
            hybrid_scores[i]
        ))

    # Prompt Assembly
    prompt = make_inference_prompt(query, similar_cases)

    # Inference
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Parse Output
    match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
    if match:
        try:
            json_output = json.loads(match.group())
        except json.JSONDecodeError:
            json_output = {"error": "Invalid JSON", "raw": raw_output}
    else:
        json_output = {"error": "No JSON found", "raw": raw_output}

    # Save Result
    result = {
        "instruction": query,
        "expected_output": item["output"],
        "model_output": json_output,

        "similar_problem_1": similar_cases[0][0],
        "similar_solution_1": similar_cases[0][1],
        "urgency_1": similar_cases[0][2],
        "category_1": similar_cases[0][3],
        "similarity_1": similar_cases[0][4],

        "similar_problem_2": similar_cases[1][0],
        "similar_solution_2": similar_cases[1][1],
        "urgency_2": similar_cases[1][2],
        "category_2": similar_cases[1][3],
        "similarity_2": similar_cases[1][4],

        "similar_problem_3": similar_cases[2][0],
        "similar_solution_3": similar_cases[2][1],
        "urgency_3": similar_cases[2][2],
        "category_3": similar_cases[2][3],
        "similarity_3": similar_cases[2][4],

        "avg_similarity": np.mean([similar_cases[0][4], similar_cases[1][4], similar_cases[2][4]])
    }

    results.append(result)

# Save results 
def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

with open("rag_test_results.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, default=convert_np, ensure_ascii=False) + "\n")

df = pd.DataFrame(results)
df["model_output"] = df["model_output"].apply(json.dumps)
df.to_csv("rag_test_results.csv", index=False)

print("All outputs saved to:")
print("- rag_test_results.jsonl")
print("- rag_test_results.csv")
