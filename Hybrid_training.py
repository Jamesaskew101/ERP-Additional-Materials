import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Load in data and pathways
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = "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"
TRAIN_FILE = "train1_final.jsonl"
TEST_FILE = "test_clean.jsonl"   

#Load in Tokenizer and Model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)

#Add LoRA adapters 

lora_config = LoraConfig(
    r=16,                    
    lora_alpha=32,           
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

#HYBRID TWEAK: Unlike pure LoRA, we also unfreeze `lm_head` and all `norm` layers to improve output adaptation and control.
for name, param in model.named_parameters():
    if "lm_head" in name or "norm" in name:
        param.requires_grad = True  # make these layers trainable

#Load Datasets
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
test_ds = load_dataset("json", data_files=TEST_FILE, split="train")

#Generating prompt
instruction_prompt = """### Instruction:
{}

### Response:
{}
"""
EOS_TOKEN = tokenizer.eos_token

def format_prompt(example):
    return {
        "text": instruction_prompt.format(example["instruction"], example["output"]) + EOS_TOKEN
    }

train_ds = train_ds.map(format_prompt)
test_ds = test_ds.map(format_prompt)

#Tokenize Datasets
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)

#Training Arguments (Hybrid fine‑tuning)
training_args = TrainingArguments(
    output_dir="outputs_hybrid",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,
    weight_decay=0.1,
    bf16=True,
    eval_strategy="steps",
    eval_steps=60,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs_hybrid",
    logging_steps=10,
    logging_first_step=True,
    report_to="none",
    remove_unused_columns=False
)

#Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test   
)

#Training
trainer_stats = trainer.train()
print("Training complete.")

#Save Model & Tokenizer
save_dir = "outputs_hybrid/final_model"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Hybrid model and tokenizer saved to {save_dir}")