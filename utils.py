def promptingBusiness(row, type):

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
    else:
        raise Exception('prompting type is not yet implemented')
    