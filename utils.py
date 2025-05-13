import pandas as pd
import numpy as np
import scipy.stats as stats

def generateConfidenceScore(csvPath):
    df = pd.read_csv(csvPath)

    idx_prompt_map = {
        0:'dag',
        1:'dg',
        2:'ag',
    }

    score_dict = {}

    for ix in range(3):
        score_dict[ix] = []
        for col in df.columns:
            if col == 'prompt': continue
            score_dict[ix].append(float(df.iloc[ix,int(col)]))

    res_list = []

    for ix in range(3):
        mean = np.mean(score_dict[ix])
        sem = stats.sem(score_dict[ix])
        confidence = 0.95
        n = len(score_dict[ix])
        df = n - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, df)
        margin_of_error = t_critical * sem
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        res_list.append((mean,lower_bound,upper_bound))
    
    # print(res_list)

    with open(csvPath.split('.csv')[0]+'.txt','w') as f:
        for ix in range(3):
            f.write(f"{idx_prompt_map[ix]}'s avg is {res_list[ix][0]} and 95% confidence intervel is {res_list[ix][1]} <-> {res_list[ix][2]}\n")

def generateWordMap(row):
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
    
    return word_map

def gen_prompts(row, prompt, word_map=None):

  if prompt == 'dg' or prompt == 'dag':
    word_map = generateWordMap(row)

  if prompt == 'refless':
    return f"""
    You are a professional machine translation evaluator.
    You will be given:
    - A sentence in Bengali, containing words in the Sylheti dialect.
    - A machine-translated English sentence.
    After evaluating whether the English sentence is an accurate translation of the Bengali source,
    your task is to provide only a score from 0 to 100. The score may be a floating-point number.
    """
  
  elif prompt == 'ag':
    return f"""
    You are a professional machine translation evaluator.
    You will be given:
    - A sentence in Bengali, containing words in the Sylheti dialect.
    - A machine-translated English sentence.
    After evaluating whether the English sentence is an accurate translation of the Bengali source,
    your task is to provide only a score from 0 to 100. The score may be a floating-point number.
    For your convenience, we are providing the scoring criteria:
    - Scores of 0–30 indicate that the translation is mostly unintelligible, 
      either completely inaccurate or containing very few words from the source sentence.
    - Scores of 31–50 suggest partial intelligibility, 
      with some words from the source sentence included but numerous grammatical errors.
    - Scores of 51–70 mean the translation is generally clear, 
      with most words from the source sentence preserved and only minor grammatical issues.
    - Scores of 71–90 indicate the translation is clear and intelligible, 
      with nearly all words from the source sentence included and only minor non-grammatical issues.
    - Scores of 91–100 reflect a perfect or near-perfect translation that accurately conveys the source meaning without any errors.
    """

  elif prompt == 'dg':
    header = f"""
    You are a professional machine translation evaluator.
    You will be given:
    - A list of Bengali words influenced by the Sylheti dialect, 
      along with their English translations, to help you better understand the dialect.
    - A sentence in Bengali that includes words from the Sylheti dialect.
    - A machine-translated English sentence.
    After evaluating whether the English sentence is an accurate translation of the Bengali source,
    your task is to provide only a score from 0 to 100. The score may be a floating-point number.
    For your reference, here is a list of Bengali words influenced by the Sylheti dialect, 
    along with their English translations:
    """
    word_list = "\n".join([f"- '{k}' translate to '{v}'" for k, v in word_map.items()])
    return header + word_list

  elif prompt == 'dag':
    header = f"""
    You are a professional machine translation evaluator.
    You will be given:
    - A list of Bengali words influenced by the Sylheti dialect, 
      along with their English translations, to help you better understand the dialect.
    - A sentence in Bengali that includes words from the Sylheti dialect.
    - A machine-translated English sentence.
    After evaluating whether the English sentence is an accurate translation of the Bengali source,
    your task is to provide only a score from 0 to 100. The score may be a floating-point number.
    For your convenience, we are providing the scoring criteria:
    - Scores of 0–30 indicate that the translation is mostly unintelligible, 
      either completely inaccurate or containing very few words from the source sentence.
    - Scores of 31–50 suggest partial intelligibility, 
      with some words from the source sentence included but numerous grammatical errors.
    - Scores of 51–70 mean the translation is generally clear, 
      with most words from the source sentence preserved and only minor grammatical issues.
    - Scores of 71–90 indicate the translation is clear and intelligible, 
      with nearly all words from the source sentence included and only minor non-grammatical issues.
    - Scores of 91–100 reflect a perfect or near-perfect translation that accurately conveys the source meaning without any errors.
    For your reference, here is a list of Bengali words influenced by the Sylheti dialect, 
    along with their English translations:
    """
    word_list = "\n".join([f"- '{k}' translate to '{v}'" for k, v in word_map.items()])
    return header + word_list


if __name__ == '__main__':
  generateConfidenceScore(csvPath='MTEresults - llama2FTcusTokRegHead.csv')