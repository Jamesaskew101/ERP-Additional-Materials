# Retriever Data

This folder contains the training data and scripts used to build the **retrieval component** of the support ticket triage system.  
The retriever is responsible for encoding problem descriptions into dense vector embeddings so that similar issues can be efficiently retrieved and matched against past resolutions.

---

## 📂 Contents
- `train.csv` → Cleaned dataset containing support ticket problems, resolutions, urgency codes, and categories.  
- `build_retriever_data.py` (example name for your script) → Script that:
  - Cleans and normalizes problem text.  
  - Generates sentence embeddings using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.  
  - Saves outputs into:
    - `embeddings.npy` → Dense vector representations of ticket problems.  
    - `metadata.parquet` → Metadata (ticket ID, problem text, resolution summary, urgency code, category).  
    - `retriever_debug_sample.csv` → A small sample of 50 rows for quick inspection.  

---

## ⚙️ Retriever Architecture (brief)
The retriever follows a **bi-encoder architecture**, where problem texts are encoded into dense vectors using a transformer-based encoder (Sentence-BERT).  
During inference, similarity is computed between the query embedding and stored embeddings (e.g. via cosine similarity) to fetch the most relevant past tickets.  
This allows efficient retrieval in downstream hybrid RAG and fine-tuning pipelines.

---

## 🚀 Usage
To build embeddings from `train.csv`:
```bash
python build_retriever_data.py
