import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import re

from utils import promptingBusiness


model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name, 
  torch_dtype=torch.bfloat16, 
  device_map="auto",
  offload_folder='offload')
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


def extract_score(text):
  match = re.search(r"\bscore\b(?:\s+\w+){0,10}?\s+(\d{1,3}(?:\.\d+)?)\b", text)
  if match:
      return float(match.group(1))
  return None


def evaluate(test_data_path, prompt):

  model_outputs = []
  da_scores = []
  cnt_none_score = 0

  df = pd.read_csv(test_data_path)
  for index, row in df.iterrows():
    content = promptingBusiness(row=row,type=prompt)
    # print(content)

    messages = [
      {"role": "user", "content": content},
      {"role": "user", "content": f"Bengali Source Sentence is: {row['src']}"},
      {"role": "user", "content": f"Machine Translated English Sentence is: {row['mt']}"}
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    
    # Added attention_mask since attention_mask cannot be inferred from input because pad token is same as eos token.
    input_tensor = {
      "input_ids": input_ids.to(model.device),
      "attention_mask": (input_ids != tokenizer.pad_token_id).long().to(model.device)
    }

    # Use both input_ids and attention_mask in model.generate
    outputs = model.generate(
      **input_tensor,
      max_new_tokens=100
    )


    result = tokenizer.decode(outputs[0][input_tensor["input_ids"].shape[1]:], skip_special_tokens=True)
    # print(result)
    outputed_score = extract_score(result)
    if outputed_score is None:
      cnt_none_score+=1
      print(f'Regex could not extract score. None score Count is {cnt_none_score}')
      model_outputs.append(50.0)
      da_scores.append(float(row['score']))
    else:
      model_outputs.append(outputed_score)
      da_scores.append(float(row['score']))
      
    
    if len(model_outputs) > 3: 
      with open('deepseekResults.txt','w') as f:
        f.write(f"Pearson : {pearsonr(model_outputs, da_scores)[0]}\n")
        f.write(f"Spearman : {spearmanr(model_outputs, da_scores)[0]}")

  print("Pearson:", pearsonr(model_outputs, da_scores)[0])
  print("Spearman:", spearmanr(model_outputs, da_scores)[0])


if __name__ == '__main__':

  evaluate(test_data_path='test_comet_da_scaled.csv', prompt='deepseekDa')
