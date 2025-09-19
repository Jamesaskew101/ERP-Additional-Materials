import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# ---------- Config ----------
CSV_PATH = "train.csv"

# Only embed problem-focused columns
EMBED_COLS = ["CLEAN_PROBLEM"]

# Metadata columns to save
META_COLS = ["TICKETID", "CLEAN_PROBLEM", "RESOLUTION_SUMMARY", "URGENCYCODE", "CATEGORY"]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("./retriever_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- Load CSV ----------
df = pd.read_csv(CSV_PATH)

# Ensure all metadata columns exist and are clean
for col in META_COLS:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str).str.strip()

# ---------- Drop rows with missing PROBLEM or SOLUTION ----------
before = len(df)
df = df[(df["CLEAN_PROBLEM"].str.len() > 0) & (df["RESOLUTION_SUMMARY"].str.len() > 0)].reset_index(drop=True)
print(f"✅ After filtering PROBLEM & SOLUTION, kept {len(df)} rows out of {before}")

# ---------- Text Normalization ----------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

# ---------- Build text for embedding ----------
def build_embed_text(row):
    parts = []
    for col in EMBED_COLS:
        val = row[col]
        if val and isinstance(val, str) and val.strip():
            parts.append(normalize_text(val))
    return " | ".join(parts)

df["__embed_text__"] = df.apply(build_embed_text, axis=1)
df = df[df["__embed_text__"].str.len() > 0].reset_index(drop=True)

# ---------- Add domain tokenized terms ----------
def fast_tokenize(text):
    return set(re.findall(r'\b\w+\b', str(text).upper()))

df["tokens"] = df["CLEAN_PROBLEM"].apply(fast_tokenize)

# --- Debug sample output ---
print("\n=== Sample metadata with embedding text and tokens ===")
print(df[META_COLS + ["__embed_text__", "tokens"]].head(5).to_string())

# Optional: save debug sample
df[META_COLS + ["__embed_text__", "tokens"]].head(50).to_csv("retriever_debug_sample.csv", index=False)
print("✅ Wrote first 50 rows to retriever_debug_sample.csv for inspection")

# ---------- Generate Embeddings ----------
print("\n🔍 Generating sentence embeddings...")
model = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = model.encode(
    df["__embed_text__"].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# ---------- Save Outputs ----------
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
df[META_COLS + ["__embed_text__", "tokens"]].to_parquet(OUTPUT_DIR / "metadata.parquet", index=False)

print(f"\n✅ Saved {embeddings.shape[0]} vectors ({embeddings.shape[1]} dims) to {OUTPUT_DIR}")
