import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from utils import gen_prompts

class MTQualityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, prompt, max_length=1024, is_llama2=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.is_llama2 = is_llama2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.is_llama2 == False:
          messages = [
              {"role": "user", "content": gen_prompts(row, self.prompt)},
              {"role": "user", "content": f"Bengali Source Sentence: {row['src']}"},
              {"role": "user", "content": f"Machine Translated English Sentence: {row['mt']}"}
          ]
        else:
          messages = [
              {
                "role": "user", 
                "content": gen_prompts(row, self.prompt) +
                f"\nBengali Source Sentence: {row['src']}" +
                f"\nMachine Translated English Sentence: {row['mt']}"
              },
              {"role": "assistant", "content": ""}
          ]
        encodings = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        try:
          return {
              "input_ids": encodings.squeeze(0),
              "attention_mask": (encodings != self.tokenizer.pad_token_id).long().squeeze(0),
              "labels": torch.tensor(float(row['score']), dtype=torch.float)
          }
        except Exception as e:
          print("apply_chat_template of the corresponding tokenizer is not behaving right.")
          exit()

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

        val_pear = evaluate(val_loader, model, device) 
        if val_pear > best_pear:
          best_pear = val_pear
          torch.save(model.regression_head.state_dict(), output_path)

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

  return pearsonr(preds, labels)[0]



def test(test_loader, model, device):
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
      custom_max_length = 1024
      is_llama2 = True
    elif model_type == 'deepseek':
      model_name = 'deepseek-ai/deepseek-llm-7b-chat'
      custom_tokenizer_name = 'deepseek-sylheti-bpe-tokenizer'
      custom_max_length = 1024
      is_llama2 = False
    elif model_type == 'openchat':
      model_name = 'openchat/openchat-3.5-1210'
      custom_tokenizer_name = 'openchat-sylheti-bpe-tokenizer'
      custom_max_length = 1024
      is_llama2 = False

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

    train_dataset = MTQualityDataset(train_df, tokenizer, prompt, custom_max_length, is_llama2)
    val_dataset = MTQualityDataset(val_df, tokenizer, prompt, custom_max_length, is_llama2)
    test_dataset = MTQualityDataset(test_df, tokenizer, prompt, custom_max_length, is_llama2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if only_test == False:
      train(train_loader, val_loader, output_path, model, device, epochs=epochs)
    test(test_loader, model, device)
