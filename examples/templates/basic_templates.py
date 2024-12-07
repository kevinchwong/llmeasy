# Templates for basic_usage.py

STORY_TEMPLATE = """
Write a short ${style} story about ${subject}.
Keep it under ${word_count} words.
"""

SENTENCE_ANALYSIS_TEMPLATE = """
Analyze the following sentence and provide the analysis in this JSON format:
{
    "subject": "main subject of the sentence",
    "verb": "main verb",
    "objects": ["list", "of", "objects"],
    "adjectives": ["list", "of", "adjectives"]
}

Sentence: ${sentence}
""" 