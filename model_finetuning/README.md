# Model Fine-Tuning

This folder contains the fine-tuning pipeline and analysis for the MSc ERP project:  
**"Triage System for Support Tickets Using LLMs"**.

The objective of this stage is to compare two parameter-efficient fine-tuning approaches — **LoRA** and a **Hybrid method** — applied to the cleaned support ticket dataset.

---

## Workflow Overview

1. **Data Preparation**
   - `Generate_test_data.py`  
     Splits the preprocessed dataset into training and test sets (95:5 split).  
     - Outputs:
       - `train.jsonl`
       - `test.jsonl`

2. **Fine-Tuning**
   - `llora_tuning.py` → Runs LoRA fine-tuning  
   - `hybrid_tuning.py` → Runs Hybrid fine-tuning  

   Both scripts were executed on the **CSF cluster** using **two A100 GPUs in parallel**.  
   Due to GPU requirements, these scripts **cannot be run locally** on standard devices.

   - Outputs:
     - `llora_tuning_results.txt`, `llora_training_results.txt`
     - `hybrid_tuning_results.txt`, `hybrid_training_results.txt`

3. **Comparative Analysis**
   - `fine_tuning_analysis.ipynb`  
     Parses the results from both LoRA and Hybrid fine-tuning runs.  
     Produces plots and metrics comparing:
     - Training loss
     - Validation loss
     - Accuracy trends

---

## Files in This Folder

- **Data Split**
  - `Generate_test_data.py`
  - `train.jsonl`
  - `test.jsonl`

- **Fine-Tuning**
  - `llora_tuning.py`
  - `Hybrid_tuning.py`
  - `llora_tuning_results.txt`, 
  - `hybrid_tuning_results.txt`,

- **Analysis**
  - `fine_tuning_analysis.ipynb`

---

## Notes

- Training was performed on the **CSF (Computational Shared Facility)**, University cluster.  
- Both methods used **Meta-LLaMA 3.1** as the base model.  
- Fine-tuning experiments were executed under identical conditions for fair comparison.  
- Local reproduction is limited to the analysis stage (`fine_tuning_analysis.ipynb`) as training requires GPU resources.
- The model we proceed with is the generated through 'hybrid training'.
- The llora fine tuned model is just there for comparaitive purposes.

