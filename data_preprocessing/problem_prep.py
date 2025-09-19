import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

#Model configuration
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = "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"

#Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    model.eval()

except Exception as e:
    print("Failed to load model or tokenizer.")
    raise e

# Creating problem prompt to extract main problem a user is having without waffle, such as introductions.
def make_prompt(problem_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful support assistant."},
        {"role": "user", "content": f"""Here is a problem from a support ticket:

{problem_text.strip()}

Please rewrite the actual technical issue clearly in plain English, without greetings, thanks, or extra commentary. Keep it under 80 words."""}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

#Inference Function
@torch.inference_mode()
def extract_problem(problem_text: str, max_new_tokens: int = 80) -> str:
    prompt = make_prompt(problem_text)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return decoded

# === Main Script: Load CSV, Clean, Save ===
def main():
    input_path = "Ticket_file_clean.csv"
    out_path = "Ticket_clean_problems.csv"

    df = pd.read_csv(input_path, dtype={"TICKETID": str})

    if "TICKETID" not in df.columns or "PROBLEM" not in df.columns:
        raise ValueError("Input file must contain columns: 'TICKETID' and 'PROBLEM'.")

    cleaned = []
    for _, row in df.iterrows():
        ticket_id = str(row["TICKETID"])
        raw_problem = row["PROBLEM"] if pd.notna(row["PROBLEM"]) else ""
        problem_text = extract_problem(raw_problem)
        cleaned.append({"TICKETID": ticket_id, "CLEAN_PROBLEM": problem_text})

    out_df = pd.DataFrame(cleaned, columns=["TICKETID", "CLEAN_PROBLEM"])
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
