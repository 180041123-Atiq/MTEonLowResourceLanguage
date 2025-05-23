import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os

from utils import gen_prompts,promptingBusiness


class DADataset(Dataset):
    def __init__(self, df, tokenizer, type='referenced', max_length=512):
        self.texts = [promptingBusiness(row=row, type=type) for _, row in df.iterrows()]
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
    

class RegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        masked = hidden_states * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.linear(pooled).squeeze(-1)


class ModelWithRegression(nn.Module):
    def __init__(self, base_model, regression_head):
        super().__init__()
        self.model = base_model
        self.regression_head = regression_head


def train(train_loader, val_loader, output_path, model, device, epochs=1, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.regression_head.parameters(), lr=lr)

    best_pear = -100.0

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                preds = model.regression_head(last_hidden, attention_mask)
                loss = nn.MSELoss()(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # val_pear = evaluate(val_loader, model, device) 
        # print(f"validation score in pear : {val_pear}")
        # if val_pear > best_pear:
        #   best_pear = val_pear
        #   try:
        #     torch.save(model.regression_head.state_dict(), output_path)
        #   except Exception as e:
        #     print(f"Could not save the regression head due to {e}")

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")



def evaluate(val_loader, model, device):
  model.eval()
  preds, labels = [], []

  with torch.no_grad():
      for batch in tqdm(val_loader, desc='Evaluating'):
          input_ids = batch["input_ids"].to(device)
          attention_mask = batch["attention_mask"].to(device)
          labels_batch = batch["labels"].to(device)

          outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
          last_hidden = outputs.hidden_states[-1]
          preds_batch = model.regression_head(last_hidden.float(), attention_mask)

          preds.extend(preds_batch.cpu().numpy())
          labels.extend(labels_batch.cpu().numpy())
  # print(f'valScore {pearsonr(preds, labels)[0]}')
  return pearsonr(preds, labels)[0]



def test(test_loader, output_path, model, device):
    
    if os.path.exists(output_path):
      model.regression_head.load_state_dict(torch.load(output_path))
    
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
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


def main(model_type, prompt, epochs, batch_size, lr, train_path, val_path, test_path, output_path, only_test, quantized, cusTok):
    
    if model_type == 'llama2':
      model_name = 'meta-llama/Llama-2-7b-chat-hf'
      custom_tokenizer_name = 'llama2-sylheti-bpe-tokenizer'
      custom_max_length = 512
    elif model_type == 'llama213b':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
        custom_tokenizer_name = 'llama213b-sylheti-bpe-tokenizer'
        custom_max_length = 512
    elif model_type == 'openchat':
      model_name = 'openchat/openchat-3.5-1210'
      custom_tokenizer_name = 'openchat-sylheti-bpe-tokenizer'
      custom_max_length = 512
    elif model_type == 'gemma':
        model_name = 'google/gemma-7b'
        custom_tokenizer_name = 'gemma-sylheti-bpe-tokenizer'
        custom_max_length = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cusTok == False:
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    if quantized == False:
      base_model = AutoModel.from_pretrained(
        model_name, 
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)
    else:
      base_model = AutoModel.from_pretrained(
        model_name, 
        output_hidden_states=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True)

    # Freeze the base model
    for param in base_model.parameters():
        param.requires_grad = False

    regression_head = RegressionHead(hidden_size=base_model.config.hidden_size).to(device)
    model = ModelWithRegression(base_model, regression_head)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_dataset = DADataset(train_df, tokenizer, prompt, custom_max_length)
    val_dataset = DADataset(val_df, tokenizer, prompt, custom_max_length)
    test_dataset = DADataset(test_df, tokenizer, prompt, custom_max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if only_test == False:
      train(train_loader, val_loader, output_path, model, device, epochs=epochs)
    test(test_loader, output_path, model, device)
