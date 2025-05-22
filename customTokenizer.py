import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import argparse

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

def trainTokenizer(model_name, dataPath):

    if model_name == 'llama2':
        model_spec = 'meta-llama/Llama-2-7b-chat-hf'
    elif model_name == 'llama213b':
        model_spec = 'meta-llama/Llama-2-13b-chat-hf'
    elif model_name == 'openchat':
        model_spec = 'openchat/openchat-3.5-1210'
    elif model_name == 'gemma':
        model_spec = 'google/gemma-7b'

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=dataPath,
        vocab_size=1000,  # You can increase this
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(model_spec)
    new_tokens = [
        token for token in tokenizer.get_vocab().keys() if token not in hf_tokenizer.get_vocab()
    ]
    hf_tokenizer.add_tokens(new_tokens)
    hf_tokenizer.save_pretrained(f"{model_name}-sylheti-bpe-tokenizer/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument For Machine Translation Evaluation Tasks")
    parser.add_argument('--model', type=str, required=True, help='name of the model to use')
    parser.add_argument('--data', type=str, required=True, help='name of the model to use')
    args = parser.parse_args()

    tokenizerDataPath = genTXT(dataPath = args.data)

    trainTokenizer(model_name=args.model,dataPath = tokenizerDataPath)