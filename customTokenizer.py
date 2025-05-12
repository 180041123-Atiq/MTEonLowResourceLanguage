import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

def genTXT(dataPath):

    tokenizerDataPath = "tokenizerData.txt"
    if os.path.exists(tokenizerDataPath):
        return tokenizerDataPath

    df = pd.read_csv(dataPath)
    sentences = df["src"].dropna().astype(str).tolist()

    with open(tokenizerDataPath, "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(line.strip() + "\n")

    return tokenizerDataPath

def trainTokenizer(dataPath):

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=dataPath,
        vocab_size=1000,  # You can increase this
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    new_tokens = [
        token for token in tokenizer.get_vocab().keys() if token not in hf_tokenizer.get_vocab()
    ]
    hf_tokenizer.add_tokens(new_tokens)
    hf_tokenizer.save_pretrained("llama2-sylheti-bpe-tokenizer/")

def trainDeepseekTokenizer(dataPath):
  tokenizer = ByteLevelBPETokenizer()
  tokenizer.train(
      files=dataPath,
      vocab_size=1000,  # You can increase this
      min_frequency=2,
      special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
  )

  # Load the tokenizer for the DeepSeek base model
  hf_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

  # Add new tokens that are not already in the original tokenizer
  new_tokens = [
      token for token in tokenizer.get_vocab().keys() if token not in hf_tokenizer.get_vocab()
  ]
  hf_tokenizer.add_tokens(new_tokens)

  # Save the extended tokenizer
  hf_tokenizer.save_pretrained("deepseek-sylheti-bpe-tokenizer/")

if __name__ == '__main__':
    tokenizerDataPath = genTXT(dataPath = 'train_comet_da_scaled.csv')
    trainDeepseekTokenizer(dataPath = tokenizerDataPath)