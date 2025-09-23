# Example Usage and Final Model

This folder demonstrates how to run the **fine-tuned hybrid LLaMA 3.2 model** and evaluate results on example queries.  
It provides a minimal pipeline that:  
1. Loads the fine-tuned model.  
2. Reads queries from a text file.  
3. Generates predictions using the retriever + generator setup.  
4. Saves outputs to a CSV file for inspection.  

---

## Files

- **`model_usage.py`**  
  Main script for running the model.  
  - Loads the fine-tuned LLaMA 3.2 model with LoRA adapters.  
  - Retrieves top-3 similar support cases using embeddings.  
  - Generates a predicted **Urgency**, **Category**, and **Resolution** for each query.  

- **`queries.txt`**  
  A plain text file containing one query per line.  
  - There are currentley three in there but more can be added.

- **`rag_txt_results.csv`**  
  Output file produced by `model_usage.py`.  
  - Contains the model’s predictions for each query in `queries.txt`.  
  - Each row includes the input query, retrieved cases with a similarity score, and generated resolution.  

---



