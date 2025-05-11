import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import get_scheduler
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from utils import promptingBusiness

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "deepseek-ai/deepseek-llm-7b-chat"
customTokenizerPath = 'deepseek-sylheti-bpe-tokenizer'
tokenizer = AutoTokenizer.from_pretrained(customTokenizerPath)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
  model_name, 
  quantization_config=bnb_config, 
  device_map="auto",
  offload_folder='offload')
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

class TranslationRegressionDataset(Dataset):
  def __init__(self, tokenizer, data_path, prompt, max_length=512):
    self.tokenizer = tokenizer
    df = pd.read_csv(data_path)
    self.texts = [
      promptingBusiness(row=row, type=prompt) for _, row in df.iterrows()
    ]
    self.scores = df['score'].values.astype('float16')
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    encoded = self.tokenizer(
        self.texts[idx],
        padding="max_length",
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "score": torch.tensor(self.scores[idx])
    }

class RegressionHead(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.linear = nn.Linear(hidden_size, 1)

  def forward(self, hidden_states, attention_mask):
    masked = hidden_states * attention_mask.unsqueeze(-1)
    pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    return self.linear(pooled).squeeze(-1)

def train(data_path, prompt, epochs):

  for param in model.parameters():
    param.requires_grad = False

  model.regression_head = RegressionHead(model.config.hidden_size).to(device)

  optimizer = torch.optim.AdamW(model.regression_head.parameters(), lr=1e-4)

  dataset = TranslationRegressionDataset(tokenizer, data_path, prompt)
  loader = DataLoader(dataset, batch_size=4, shuffle=True)

  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      targets = batch["score"].to(device)

      with torch.cuda.amp.autocast():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        preds = model.regression_head(last_hidden, attention_mask)
        loss = nn.MSELoss()(preds, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")


def evaluate(data_path, prompt):

  dataset = TranslationRegressionDataset(tokenizer, data_path, prompt)
  test_loader = DataLoader(dataset, batch_size=4, shuffle=True)

  model.eval()
  preds, labels = [], []

  with torch.no_grad():
    for batch in test_loader:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels_batch = batch["score"].to(device)

      outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
      last_hidden = outputs.hidden_states[-1]
      preds_batch = model.regression_head(last_hidden.float(), attention_mask)

      preds.extend(preds_batch.cpu().numpy())
      labels.extend(labels_batch.cpu().numpy())

  print("Pearson:", pearsonr(preds, labels)[0])
  print("Spearman:", spearmanr(preds, labels)[0])