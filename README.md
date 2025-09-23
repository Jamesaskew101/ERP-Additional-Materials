# ERP-Additional-Materials
James Askew- 10857809, additional materials

This project makes use of Meta’s **LLaMA 3.2** model under the 
[LLaMA 3.2 Community License Agreement](https://www.llama.com/llama-downloads/).

A full summary of licensing terms, attribution requirements, and the 700M MAU clause 
can be found in the [LICENSE README](LICENSE/README.md).

## Repository Structure

This repository is organised into the following main components:

- **`data_preprocessing/`**  
  Contains scripts and notebooks for cleaning and preparing the raw support ticket data.  
  - Handles normalisation of text, and preparation of overall data.
  - This also extracts clean solution and problem process using a raw LLaMA 3.2 model, needing high computational resources to run.  
  - Outputs structured datasets (not included here due to confidentiality).  

- **`model_finetuning/`**  
   Provides the infulstructure for model fine tuning.
  - Uses two methods of fine-tuning for comparitive analysis.
  - Final model fine-tuning to be used is that of the Hybrid_tuning.py.
  - Analyses training results using fine_tuning_analysis.ipynb.
  - Output is the trained model (final_model), avaliable apon request.

- **`model_analysis/`**  
  Provides scripts for evaluating model performance comparing to raw model.
  - Tests model on test.jsonl.
  - Computes embedding similarity, retrieval scores, and other evaluation metrics all stored in large file with retrieved cases (hybrid_results.csv).  
  - model_analysis.ipynb includes analysis of overall metrics as well as a deeper analysis of.  

- **`retriever_data/`**  
  Contains the process of generating embeddings and metadata used for semantic retrieval.  
  - Built with `sentence-transformers/all-MiniLM-L6-v2`.  
  - Enables retrieval-augmented generation (RAG) by matching new queries to past tickets.  
  - The embeddings here are placeholders; full retriever data is excluded due to confidentiality.
  

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
  [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) or [Meta’s official site](https://www.llama.com/llama-downloads/).  
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

## Reproduction of final model
First access to model via hugging face is needed. then need to request access to data. 
### Data Preperation 
In the data_preprocessing folder first run data_prep1.ipynb, followed by problem_prep.py and solution_prep.py to give a output of final_output.csv.

### Model Training
Pass this final_output.csv through Generate_test_data.py. Then run the Hybrid_tuning.py

### Retriver Data
- Then run the retriver_embeddings_generation.py which will then create embeddings for the retrival process

### Final Model
- Final model is then ready to run, recomended implementation can be seen in the Example usage and final model folder.

## Environment Setup

This project was developed and tested with:
- Python 3.10
- PyTorch 2.8.0
- Hugging Face Transformers 4.56.1
- Datasets 4.1.0
- Sentence-Transformers 5.1.0
- scikit-learn 1.7.2

All required dependencies are listed in [requirements.txt](requirements.txt).

To create the environment:

```bash
conda create --name llama3train python=3.10
conda activate llama3train
pip install -r requirements.txt
```


## Hardware Requirements

- Fine-tuning and hybrid training were performed on a single **NVIDIA A100 GPU (40GB VRAM)**.  
- The fine-tuned LLaMA 3.2 model (with LoRA adapters) requires approximately **8–10 GB VRAM for inference** and around **40 GB VRAM for full fine-tuning**.  
- Model checkpoint size:  
  - Base LLaMA 3.2 model: ~8B parameters  
  - Fine-tuned LoRA adapter weights: ~2–3 GB



