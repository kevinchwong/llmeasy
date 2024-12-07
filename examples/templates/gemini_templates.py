ANALYSIS_TEMPLATE = """
Analyze the following text and provide insights in JSON format:
{
    "main_themes": ["theme1", "theme2"],
    "key_points": ["point1", "point2"],
    "sentiment": "overall sentiment",
    "recommendations": ["rec1", "rec2"]
}

Text: ${text}
"""

CREATIVE_TEMPLATE = """
Write a ${style} piece about ${topic} in approximately ${length}.
"""

TECHNICAL_TEMPLATE = """
Explain ${concept} to ${audience} using a ${format}.
""" 