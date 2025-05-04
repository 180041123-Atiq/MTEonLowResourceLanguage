import pandas as pd

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

def promptingBusiness(row, type, word_map=None):

    if type == 'dg' or type == 'dag' or 'deepseek' in type:
        word_map = generateWordMap(row)

    if type == 'referenced':
        return f"""<s>[INST] <<SYS>>
        We need to Evaluate the machine translated senetences of Bengali with Sylheti dialect to English (Translation),
        with quality scores. Predict the quality score in the range of 0 to 100 considering the following source, machine 
        translated and referenced texts. Predict only the score, no need for explanation.
        <<SYS>>
        Source: {row['src']}
        Translation: {row['mt']}
        Reference: {row['ref']}
        Score: [/INST]"""
    
    elif type == 'referenceless':
        return f"""<s>[INST] <<SYS>>
        We need to Evaluate the machine translated senetences of Bengali with Sylheti dialect to English (Translation),
        with quality scores. Predict the quality score in the range of 0 to 100 considering only the source and 
        machine translated texts. Predict only the score, no need for explanation.
        <<SYS>>
        Source: {row['src']}
        Translation: {row['mt']}
        Score: [/INST]"""
    
    elif type == 'ag':
        return f"""<s>[INST] <<SYS>>
        We need to Evaluate the machine translated senetences of Bengali with Sylheti dialect to English (Translation),
        with quality scores ranging from 0 to 100.
        Scores of 0-30 indicate that the translation is mostly unintelligible, either completely inaccurate or 
        containing only some keywords. Scores of 31-50 suggest partial intelligibility, with some keywords present but 
        numerous grammatical errors. A score between 51-70 means the translation is generally clear, with most keywords 
        included and only minor grammatical errors. Scores of 71-90 indicate the translation is clear and intelligible,
        with all keywords present and only minor non-grammatical issues. Finally, scores of 91-100 reflect a perfect or
        near-perfect translation, accurately conveying the source meaning without errors.

        The evaluation criteria focus on two main aspects: Adequacy (how much information is conveyed) and Fluency 
        (how grammatically correct the translation is). Predict the quality score in the range of 0 to 100 considering 
        the above instructions. Predict only the score, no need for explanation.
        <</SYS>>

        Source: {row['src']}
        Translation: {row['mt']}
        Score: [/INST]"""
    
    elif type == 'ds':
        return f"""<s>[INST] <<SYS>>
        You are an expert evaluator for machine translation systems. Your task is to evaluate the quality of English translations of Bengali sentences written in the Sylheti dialect.

        Each translation should be assigned a quality score between 0 and 100, where:
        - 100 means a perfect translation (accurate and fluent)
        - 0 means a completely incorrect or unreadable translation

        To assist in your evaluation, here are some important Sylheti dialect keywords and their correct English equivalents:
        - "অতা" → "they"
        - "তারে" → "him"
        - "লাকান" → "like"
        - "খারণ" → "because"
        - "ভিত্রে" → "in"
        - "উফরে" → "on"
        - "কিতা" → "what"
        - "খে" → "who"
        - "বাদে" → "after"
        - "লগে" → "with"
        - "অউ" → "this"
        - "যে" → "that"
        - "কুনটা" → "which"
        - "অটাত" → "at"

        If these keywords are mistranslated or omitted, the score should be lowered accordingly.

        Use the following evaluation criteria:
        1. **Adequacy** - Does the translation preserve the meaning of the original?
        2. **Fluency** - Is the translation grammatically correct and natural in English?

        Predict **only** the final quality score (a number between 0 and 100). Do not provide any explanation.
        <</SYS>>

        Source: {row['src']}
        Translation: {row['mt']}
        Score: [/INST]"""
    elif type == 'dg':
        header = (
            "<s>[INST] <<SYS>>"
            "You are a professional machine translation evaluator.\n\n"
            "You will be given:\n"
            "- A sentence in Bengali, containing words from the Sylheti dialect.\n"
            "- A machine-translated English sentence.\n"
            "- A list of Bengali (Sylheti) words with their English meanings to help you understand the source better.\n\n"
            "Your task:\n"
            "- Evaluate the English translation based on how well it preserves the meaning (adequacy) and how natural the English sounds (fluency).\n"
            "- Provide a score from 0 to 100, where:\n"
            "  - 100 means perfect translation (both fluent and faithful)\n"
            "  - 0 means completely incorrect or meaningless\n\n"
            "Here are some word meanings to help you understand the dialect:\n"
        )
        word_list = "\n".join([f"- {k} → {v}" for k, v in word_map.items()])
        body = (
            "\n\nNow evaluate the following:\n\n"
            "<</SYS>>"
            f"Source (Bengali with Sylheti dialect):\n\"{row['src']}\"\n\n"
            f"Machine Translation (English):\n\"{row['mt']}\"\n\n"
            "Score (0–100):[/INST]"
        )
        return header + word_list + body
    elif type == 'dag':
        header = (
            "<s>[INST] <<SYS>>"
            "You are a professional machine translation evaluator.\n\n"
            "You will be given:\n"
            "- A sentence in Bengali, containing words from the Sylheti dialect.\n"
            "- A machine-translated English sentence.\n"
            "- A list of Bengali (Sylheti) words with their English meanings to help you understand the source better.\n\n"
            "Your task:\n"
            "- Evaluate the English translation based on how well it preserves the meaning (adequacy) and how natural the English sounds (fluency).\n"
            "- Provide a score from 0 to 100, where:\n"
            "  - Scores of 0-30 indicate that the translation is mostly unintelligible, either completely inaccurate or containing only some keywords.\n"
            "  - Scores of 31-50 suggest partial intelligibility, with some keywords present but numerous grammatical errors.\n"
            "  - A score between 51-70 means the translation is generally clear, with most keywords included and only minor grammatical errors.\n"
            "  - Scores of 71-90 indicate the translation is clear and intelligible, with all keywords present and only minor non-grammatical issues.\n" 
            "  - Finally, scores of 91-100 reflect a perfect or near-perfect translation, accurately conveying the source meaning without errors.\n\n"
            "Here are some word meanings to help you understand the dialect:\n"
        )
        word_list = "\n".join([f"- {k} → {v}" for k, v in word_map.items()])
        body = (
            "\n\nNow evaluate the following:\n\n"
            "<</SYS>>"
            f"\nSource (Bengali with Sylheti dialect):\n\"{row['src']}\"\n\n"
            f"Machine Translation (English):\n\"{row['mt']}\"\n\n"
            "Score (0–100):[/INST]"
        )
        return header + word_list + body
    
    elif type == 'deepseekDag':
      header = f"""
      You are a professional machine translation evaluator.\n
        You will be given:
        - A list of Bengali (Sylheti) words with their English translation to help you understand Sylheti dialect.
        - A sentence in Bengali, containing words with the Sylheti dialect.
        - A machine-translated English sentence.
        After evaluating the machine translated english sentence,
        your task is to only provide a score from 0 to 100 that can have floating points. And the scoring criteria is:
        - Scores of 0-30 indicate that the translation is mostly unintelligible, either completely inaccurate or containing few translated words present in source sentence.
        - Scores of 31-50 suggest partial intelligibility, with some translated words present in the source sentence but numerous grammatical errors.
        - A score between 51-70 means the translation is generally clear, with most translated words present in the source sentence and only minor grammatical errors.
        - Scores of 71-90 indicate the translation is clear and intelligible, with almost every translated words present in the source sentence and only minor non-grammatical issues.
        - Finally, scores of 91-100 reflect a perfect or near-perfect translation, accurately conveying the source meaning without errors.
        \nHere are some english translation of Bengali words with Sylheti dialect to help you better understand the dialect:"""
      word_list = "\n".join([f"- '{k}' translate to '{v}'" for k, v in word_map.items()])
      body = f"""\nImportant: No explanations or comments in your output – just a single score that can be an integer or can have floating points!
      """
      
      return header + word_list + body 

    elif type == 'deepseekDg':
      header = f"""
      You are a professional machine translation evaluator.\n
        You will be given:
        - A list of Bengali (Sylheti) words with their English translation to help you understand Sylheti dialect.
        - A sentence in Bengali, containing words with the Sylheti dialect.
        - A machine-translated English sentence.
        Evaluate the English translation based on how well it preserves the meaning (adequacy) and how natural the English sounds (fluency).
        After evaluating the machine translated english sentence,
        your task is to only provide a score from 0 to 100 that can have floating points. 
        \nHere are some english translation of Bengali words with Sylheti dialect to help you better understand the dialect:"""
      word_list = "\n".join([f"- '{k}' translate to '{v}'" for k, v in word_map.items()])
      body = f"""\nImportant: No explanations or comments in your output – just a single score that can be an integer or can have floating points!
      """
      
      return header + word_list + body

    elif type == 'deepseekAg':

      header = f"""
      You are a professional machine translation evaluator.\n
        You will be given:
        - A sentence in Bengali, containing words with the Sylheti dialect.
        - A machine-translated English sentence.
        After evaluating the machine translated english sentence,
        your task is to only provide a score from 0 to 100 that can have floating points. And the scoring criteria is:
        - Scores of 0-30 indicate that the translation is mostly unintelligible, either completely inaccurate or containing few translated words present in source sentence.
        - Scores of 31-50 suggest partial intelligibility, with some translated words present in the source sentence but numerous grammatical errors.
        - A score between 51-70 means the translation is generally clear, with most translated words present in the source sentence and only minor grammatical errors.
        - Scores of 71-90 indicate the translation is clear and intelligible, with almost every translated words present in the source sentence and only minor non-grammatical issues.
        - Finally, scores of 91-100 reflect a perfect or near-perfect translation, accurately conveying the source meaning without errors."""
      body = f"""\nImportant: No explanations or comments in your output – just a single score that can be an integer or can have floating points!
      """
      
      return header + body 

    else:
        raise Exception(f'prompting type {type} is not yet implemented')
    
