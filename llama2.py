import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

### CONFIG
model_name = "meta-llama/Llama-2-7b-chat-hf"
use_reference = True
batch_size = 2
max_length = 512
epochs = 1
lr = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

### LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

### PROMPT TEMPLATE
def create_prompt(row, use_reference=True):
  if use_reference:
    return f"""<s>[INST] <<SYS>>
    You are a helpful quality estimation assistant. Predict the quality score of a machine translation.
    <</SYS>>

    Source: {row['src']}
    Translation: {row['mt']}
    Reference: {row['ref']}
    Score: [/INST]"""
  else:
    return f"""<s>[INST] <<SYS>>
    You are a helpful quality estimation assistant. Predict the quality score of a machine translation.
    <</SYS>>

    Source: {row['src']}
    Translation: {row['mt']}
    Score: [/INST]"""

### DATASET
class DADataset(Dataset):
    def __init__(self, df, tokenizer, use_reference=True, max_length=512):
        self.texts = [create_prompt(row, use_reference) for _, row in df.iterrows()]
        self.targets = df["score"].tolist()
        self.encodings = tokenizer(self.texts, padding="max_length", truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.targets[idx], dtype=torch.float),
        }

### LOAD MODEL WITH 4-BIT
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, peft_config)
print("LoRA params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Custom regression head
class RegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        masked = hidden_states * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.linear(pooled).squeeze(-1)

# Attach regression head
model.regression_head = RegressionHead(model.config.hidden_size).to(device)

### LOAD DATA
train_df = pd.read_csv("train_comet_da.csv")
test_df = pd.read_csv("test_comet_da.csv")

train_dataset = DADataset(train_df, tokenizer, use_reference, max_length)
test_dataset = DADataset(test_df, tokenizer, use_reference, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

### TRAINING LOOP
def train(model, train_loader, epochs=1, lr=2e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                preds = model.regression_head(last_hidden, attention_mask)
                loss = nn.MSELoss()(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

### EVALUATION
def evaluate(model, test_loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            preds_batch = model.regression_head(last_hidden.float(), attention_mask)

            preds.extend(preds_batch.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    print("Pearson:", pearsonr(preds, labels)[0])
    print("Spearman:", spearmanr(preds, labels)[0])

### RUN
train(model, train_loader, epochs=epochs, lr=lr)
evaluate(model, test_loader)
