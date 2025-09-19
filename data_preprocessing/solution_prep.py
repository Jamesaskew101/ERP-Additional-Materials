import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Getting pathway and HF token from hugging face
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = "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"

# Load Tokenizer and Model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

# Create prompt asking model to extract key steps professionals took when resolving a given problem
def make_prompt(resolution_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful support assistant."},
        {"role": "user", "content": f"""Here is a resolution log from a support ticket:
{resolution_text.strip()}
Please summarise the key actions taken to resolve the issue, clearly and concisely in plain English.
Do not include the original problem, ticket numbers, greetings, or irrelevant details.
Do not use bullet points or numbers. Keep the response under 100 words."""}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# Creating Inference
@torch.inference_mode()
def generate_summary(resolution_text: str, max_new_tokens: int = 160) -> str:
    prompt = make_prompt(resolution_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

# Main script loop 
def main():
    input_path = "Ticket_query_resolution.csv"
    out_path = "Ticket_resolution_summaries.csv"

    df = pd.read_csv(input_path, dtype={"TICKETID": str})
    if "TICKETID" not in df.columns or "RESOLUTION" not in df.columns:
        raise ValueError("Input file must contain columns: 'TICKETID' and 'RESOLUTION'.")

    summaries = []
    for _, row in df.iterrows():
        ticket_id = str(row["TICKETID"])
        raw_res = row["RESOLUTION"] if pd.notna(row["RESOLUTION"]) else ""
        summary = generate_summary(raw_res)
        summaries.append({"TICKETID": ticket_id, "RESOLUTION_SUMMARY": summary})
    out_df = pd.DataFrame(summaries, columns=["TICKETID", "RESOLUTION_SUMMARY"])
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(out_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
