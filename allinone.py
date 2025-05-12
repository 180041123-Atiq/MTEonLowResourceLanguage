import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics import pearsonr, spearmanr
from tqdm import tqdm


class MTQualityDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        messages = [
            {"role": "user", "content": f"Bengali Source Sentence: {row['src']}"},
            {"role": "user", "content": f"Machine Translated English Sentence: {row['mt']}"},
            {"role": "user", "content": "Please give a DA score between 0 and 100."}
        ]
        encodings = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(row['score']), dtype=torch.float)
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    tokenizer_output = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask}, return_tensors="pt")
    return {**tokenizer_output, "labels": labels}


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


def train(train_loader, model, device, epochs=1, lr=2e-5):
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


def evaluate(test_loader, model, device):
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


if __name__ == "__main__":
    model_name = "openchat/openchat-3.5-1210"
    train_path = "train.csv"
    test_path = "test.csv"
    batch_size = 2
    epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    regression_head = RegressionHead(hidden_size=base_model.config.hidden_size).to(device)
    model = ModelWithRegression(base_model, regression_head)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_dataset = MTQualityDataset(train_df, tokenizer)
    test_dataset = MTQualityDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    train(train_loader, model, device, epochs=epochs)
    evaluate(test_loader, model, device)
