import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

from utils import promptingBusiness

# Load CSV files
train_df = pd.read_csv("train_df_comet.csv")
test_df = pd.read_csv("test_df_comet.csv")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
model_name = "openchat/openchat-3.5-1210"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if missing

# Tokenization
def tokenize(example):
    prompt = f"### Input:\n{example['source']}\n\n### Response:\n{example['target']}\n\n### Score:\n{example['score']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

# Load model and prepare for LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # set to False if using full precision
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # adjust based on architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./openchat3.5-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    peft_config=peft_config,
    max_seq_length=512
)

# Train and evaluate
def train():
    trainer.train()
def evaluate():
    trainer.evaluate()