# ERP-Additional-Materials
James Askew- 10857809, additional materials

This project makes use of Meta’s **LLaMA 3.2** model under the 
[LLaMA 3.2 Community License Agreement](https://www.llama.com/llama-downloads/).

A full summary of licensing terms, attribution requirements, and the 700M MAU clause 
can be found in the [LICENSE README](LICENSE/README.md).

## Data Availability

This project makes use of confidential ERP support ticket datasets provided by Datel.  
Due to **data confidentiality and GDPR restrictions**, the following files are **not included** in this repository:

- `Copy_of_Tickets_cleaned.xlsx`
- `Ticket history 3.xlsx`
- `final_output.csv`
- `train.csv`
- `test.csv`
- `test.jsonl`
- `train.jsonl`
- `raw_results.csv`
- `hybrid_results.csv`
- `embeddings.npy`
- `metadata.parquet`

These files contain proprietary customer support data and cannot be shared publicly.  
- **Base Model:**  
  This project builds on **Meta’s LLaMA 3.2** model, which must be downloaded separately from  
  [Hugging Face](https://huggingface.co/meta-llama) or [Meta’s official site](https://www.llama.com/llama-downloads/).  
  Access requires an approved Hugging Face account and an access token.  

- **Hugging Face Token:**  
  For security reasons, the Hugging Face token used during experimentation has been **redacted (*****)** and is **not included** in this repository.  
  Users must supply their own valid token.  

- **Fine-Tuned Model:**  
  The final fine-tuned model (including LoRA adapter weights) is **too large to be uploaded** to GitHub.  
  If examiners or collaborators require access to the model for verification, it can be provided directly upon request.  

For reproducibility:
- The repository includes **all preprocessing, training, and evaluation scripts**.  
- Data paths are maintained in the code so that users with appropriate access can reproduce results by placing the original files in the same directory structure.  
- Data can be accessed apon request. 
