import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# config from hugging face
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = "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"
TRAIN_FILE = "train1_final.jsonl"
TEST_FILE = "test_clean.jsonl"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)

# Now apply LoRa adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load train and test data files
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
test_dataset = load_dataset("json", data_files=TEST_FILE, split="train")

# Creating prompts
instruction_prompt = """### Instruction:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token

def format_prompt(example):
    return {
        "text": instruction_prompt.format(example["instruction"], example["output"]) + EOS_TOKEN
    }

train_dataset = train_dataset.map(format_prompt)
test_dataset = test_dataset.map(format_prompt)

# Tokenize the dataset
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

# Training arguments, explained and justified in hybird train file
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-6,
    weight_decay=0.01,
    bf16=True,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=60,
    logging_first_step=True,
    report_to="none",
    remove_unused_columns=False
)

# trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)

# Start data training
trainer_stats = trainer.train()
print("Training complete.")

# Save Fine-Tuned Model and Tokenizer 
save_dir = "outputs/final_model"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer (with LoRA) saved to {save_dir}")
