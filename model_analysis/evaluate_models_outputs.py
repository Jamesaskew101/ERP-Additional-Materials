import pandas as pd
import json
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
import torch
import nltk

nltk.download('punkt')

# === Load model predictions ===
df_pred = pd.read_csv("rag_test_results.csv")
df_pred['ticketID'] = df_pred['instruction'].str.extract(r"ticketID:\s*([^\n]+)")

# === Extract model outputs ===
def extract_field(json_str, key):
    try:
        return json.loads(json_str).get(key)
    except:
        return None

df_pred['pred_urgency_code'] = df_pred['model_output'].apply(lambda x: extract_field(x, "Urgency Code"))
df_pred['pred_category'] = df_pred['model_output'].apply(lambda x: extract_field(x, "Category"))
df_pred['pred_resolution'] = df_pred['model_output'].apply(lambda x: extract_field(x, "Resolution"))

# === Load ground truth ===
df_truth = pd.read_csv("james_test.csv")

# === Merge with ground truth
df_combined = df_pred.merge(
    df_truth[['TICKETID', 'CLEAN_PROBLEM', 'URGENCYCODE', 'CATEGORY', 'RESOLUTION_SUMMARY']],
    left_on='ticketID',
    right_on='TICKETID',
    how='left'
)

# === Rename ground truth fields for clarity
df_combined.rename(columns={
    "URGENCYCODE": "true_urgency_code",
    "CATEGORY": "true_category",
    "RESOLUTION_SUMMARY": "true_resolution"
}, inplace=True)

# === Fuzzy match (Resolution) ===
df_combined['fuzzy_ratio'] = df_combined.apply(
    lambda r: fuzz.token_set_ratio(str(r['true_resolution']), str(r['pred_resolution']))
    if pd.notna(r['true_resolution']) and pd.notna(r['pred_resolution']) else None,
    axis=1
)

# === Sentence-BERT embedding similarity ===
print("🔍 Computing Sentence-BERT cosine similarities...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

truth_texts = df_combined['true_resolution'].fillna("").astype(str).tolist()
pred_texts = df_combined['pred_resolution'].fillna("").astype(str).tolist()

truth_embeds = embed_model.encode(truth_texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)
pred_embeds = embed_model.encode(pred_texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

truth_embeds = torch.nn.functional.normalize(truth_embeds, p=2, dim=1)
pred_embeds = torch.nn.functional.normalize(pred_embeds, p=2, dim=1)

cos_sims = torch.sum(truth_embeds * pred_embeds, dim=1).cpu().numpy()
df_combined['resolution_embedding_similarity'] = cos_sims

# === Retriever Similarity ===
if all(col in df_combined.columns for col in ['similarity_1', 'similarity_2', 'similarity_3']):
    df_combined['avg_retriever_similarity'] = df_combined[['similarity_1', 'similarity_2', 'similarity_3']].mean(axis=1)

# === Save full CSV ===
output_file = "hybrid_results.csv"
df_combined.to_csv(output_file, index=False)

print(f"\n✅ Results saved to '{output_file}'")
