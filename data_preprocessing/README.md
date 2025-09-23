# Data Preprocessing

This document describes the preprocessing workflow for the MSc ERP project:
**"Triage System for Support Tickets Using LLMs"**.

The aim of preprocessing is to transform raw support ticket data into a structured format that can be used for model training and evaluation.

---

## Overview of Workflow

1. **Raw Data**
   - `Copy_of_Tickets_cleaned.xlsx`
   - `Ticket history 3.xlsx`

   These files are not provided due to the sensitive nature of our project. Full dataset can be accessed apon request from the Datel team or speak to Andrea Lagna.

2. **Preliminary Cleaning**
   - `data_prep1.ipynb`

   This notebook:
   - Loads the raw Excel files
   - Drops irrelevant columns
   - Applies initial cleaning (normalisation, formatting, consistency checks)
   - Produces two cleaned CSV files as output.

3. **Problem and Resolution Processing**
   - `problem_prep.py`
     - Uses the LLaMA 3 model (`meta-llama/Meta-Llama-3.1-8B-Instruct`) to rewrite messy *problem descriptions* into clear, concise statements.
   - `solution_prep.py`
     - Uses the same model to summarise *resolution logs* into short, plain-English descriptions of the steps taken to fix the issue.

4. **Final Output**
   - `final_output.csv`

   Combines the processed problems and solutions into a single structured dataset suitable for downstream tasks (e.g. training, evaluation, or building a triage system). This data file again is again hidden due data procoutions taken with this project.

---

## Files in This Folder

- **Raw Data**
  - Avaliable apon request 
  - `Copy_of_Tickets_cleaned.xlsx`
  - `Ticket history 3.xlsx`

- **Preprocessing**
  - `data_prep1.ipynb` → Notebook for initial data cleaning
  - `problem_prep.py` → Cleans and rewrites ticket "PROBLEM" fields
  - `solution_prep.py` → Summarises ticket "RESOLUTION" fields

- **Outputs**
  -  Avaliable apon request 
  - `final_output.csv` → Fully processed dataset

