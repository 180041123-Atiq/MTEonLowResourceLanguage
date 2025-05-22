import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generateConfidenceScore(csvPath):
    df = pd.read_csv(csvPath)

    idx_prompt_map = {
        0:'ag',
        1:'dg',
        2:'dag',
    }

    expn_idx = {
      'pear1':1, 
      'pear2':2, 
      'pear3':3, 
      'pear4':4, 
      'pear5':5,
      'spear1':7, 
      'spear2':8, 
      'spear3':9, 
      'spear4':10, 
      'spear5':11 
    }

    pear_score_dict = {}
    spear_score_dict = {}

    # print(df.columns)

    for ix in range(3):
        pear_score_dict[ix] = []
        spear_score_dict[ix] = []
        pii = 1
        sii = 1
        for col in df.columns:
            if col == 'prompt' or col == '95confScorePear' or col == '95confScoreSpear' : continue
        
            if f'pear{pii}' == col: 
              pii += 1
              pear_score_dict[ix].append(float(df.iloc[ix,expn_idx[col]]))
            if f'spear{str(sii)}' == col: 
              sii += 1
              spear_score_dict[ix].append(float(df.iloc[ix,expn_idx[col]]))
            # print(col)

    # print(len(spear_score_dict[0]))
    # print(len(pear_score_dict[0]))
    # exit()

    pear_res_list = []
    spear_res_list = []

    for ix in range(3):
        mean = np.mean(pear_score_dict[ix])
        sem = stats.sem(pear_score_dict[ix])
        confidence = 0.95
        n = len(pear_score_dict[ix])
        df = n - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, df)
        margin_of_error = t_critical * sem
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        pear_res_list.append((mean,lower_bound,upper_bound))

    for ix in range(3):
      mean = np.mean(spear_score_dict[ix])
      sem = stats.sem(spear_score_dict[ix])
      confidence = 0.95
      n = len(spear_score_dict[ix])
      df = n - 1
      t_critical = stats.t.ppf((1 + confidence) / 2, df)
      margin_of_error = t_critical * sem
      lower_bound = mean - margin_of_error
      upper_bound = mean + margin_of_error
      spear_res_list.append((mean,lower_bound,upper_bound))
    
    # print(res_list)

    # for ix in range(3):
    #   print(res_list[ix][0]-res_list[ix][1],res_list[ix][2]-res_list[ix][0])

    with open(csvPath.split('.csv')[0]+'.txt','w') as f:
        for ix in range(3):
            f.write(f"llama2 13B with {idx_prompt_map[ix]}'s  95% confidence intervel in pearson metric is {round(pear_res_list[ix][0],4)}+-{round(pear_res_list[ix][2]-pear_res_list[ix][0],4)}\n")
        f.write('\n')
        for ix in range(3):
          f.write(f"llama2 13B with {idx_prompt_map[ix]}'s  95% confidence intervel in Spearman metric is {round(spear_res_list[ix][0],4)}+-{round(spear_res_list[ix][2]-spear_res_list[ix][0],4)}\n")

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




def promptingBusiness(row, type, word_map=None):

    if type == 'dg' or type == 'dag' or 'deepseek' in type:
        word_map = generateWordMap(row)

    
    if type == 'refless':
        return f"""<s>[INST] <<SYS>>
        You are a professional machine translation evaluator.
        You will be given:
        - A sentence in Bengali, containing words in the Sylheti dialect.
        - A machine-translated English sentence.
        After evaluating whether the English sentence is an accurate translation of the Bengali source,
        your task is to provide only a score from 0 to 100, no need for explanation. The score may be a floating-point number.
        <<SYS>>
        Bengali Source Sentence: {row['src']}
        Machine Translated English Sentence: {row['mt']}
        Score: [/INST]"""
    
    elif type == 'ag':
        return f"""<s>[INST] <<SYS>>
        You are a professional machine translation evaluator.
        You will be given:
        - A sentence in Bengali, containing words in the Sylheti dialect.
        - A machine-translated English sentence.
        After evaluating whether the English sentence is an accurate translation of the Bengali source,
        your task is to provide only a score from 0 to 100, no need for explanation. The score may be a floating-point number.
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
        Now evaluate the following and give only the score:
        <</SYS>>
        Bengali Source Sentence: {row['src']}
        Machine Translated English Sentence: {row['mt']}
        Score: [/INST]"""
  
    elif type == 'dg':
        header = f"""<s>[INST] <<SYS>>"
          You are a professional machine translation evaluator.
          You will be given:
          - A list of Bengali words influenced by the Sylheti dialect, 
            along with their English translations, to help you better understand the dialect.
          - A sentence in Bengali that includes words from the Sylheti dialect.
          - A machine-translated English sentence.
          After evaluating whether the English sentence is an accurate translation of the Bengali source,
          your task is to provide only a score from 0 to 100, no need for explanation. The score may be a floating-point number.
          For your reference, here is a list of Bengali words influenced by the Sylheti dialect, 
          along with their English translations:
        """
        word_list = "\n".join([f"- {k} → {v}" for k, v in word_map.items()])
        body = f"""
          Now evaluate the following and give only the score:
          <</SYS>>
          Bengali Source Sentence: {row['src']}
          Machine Translated English Sentence: {row['mt']}
          Score: [/INST]
        """
        return header + word_list + body
    elif type == 'dag':
        header = f"""<s>[INST] <<SYS>>"
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
        word_list = "\n".join([f"- {k} → {v}" for k, v in word_map.items()])
        body = f"""
          Now evaluate the following and give only the score:
          <</SYS>>
          Bengali Source Sentence: {row['src']}
          Machine Translated English Sentence: {row['mt']}
          Score: [/INST]
        """
        return header + word_list + body



def genStackedBarChart():
  pipelines = ['AG', 'DG', 'DAG']
  pearsons = [0.1765, 0.279, 0.2196]

  plt.barh(pipelines, pearsons, color='tomato')  # You can use any named color or hex code

  # plt.title('Performance of Llama2 13B in SOTA pipelines')
  plt.xlabel('Pearson Correlation Metric')      # Corrected: Sales on the x-axis
  plt.ylabel('Pipelines')     # Corrected: Fruits on the y-axis
  plt.show()


if __name__ == '__main__':
  # generateConfidenceScore(csvPath='llama213bfine4.csv')
  genStackedBarChart()