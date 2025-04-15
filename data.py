from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

from utils import promptingBusiness

class DADataset(Dataset):
    def __init__(self, df, tokenizer, type='referenced', max_length=512):
        if type == 'dg':
            self.texts = [dgPrompting(row=row) for _, row in df.iterrows()]
        else:
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
    
def generateDataLoaders(train_data_path, test_data_path, tokenizer, batch_size=2, type='referenced', max_length=512):
    
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_dataset = DADataset(train_df, tokenizer, type, max_length)
    test_dataset = DADataset(test_df, tokenizer, type, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader,test_loader)

def dgPrompting(row):
    df = pd.read_csv('sylheti_dictionary.csv')
    sylheti = df['Sylheti'].tolist()
    english = df['English'].tolist()

    src = row['src']
    src_list = src.split(' ')

    word_map = {}

    for src_word in src_list:
        if src_word in sylheti:
            idx = sylheti.index(src_word)
            word_map[src_word] = english[idx]
    
    return promptingBusiness(row,'dg',word_map)




# if __name__ == '__main__':
#     dgPrompting()