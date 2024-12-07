# Templates for openai_example.py

POEM_TEMPLATE = """
Write a ${length} ${style} poem about ${subject}.
"""

REVIEW_ANALYSIS_TEMPLATE = """
Analyze this movie review and extract in this JSON format:
{
    "sentiment": "overall sentiment",
    "rating": "number between 1-5",
    "key_points": ["point1", "point2", "point3"],
    "improvements": ["improvement1", "improvement2"]
}

Review: ${review}
"""

LANGUAGE_COMPARISON_TEMPLATE = """
Compare these programming languages in this JSON format:
{
    "learning_curve": {
        "lang1": "description",
        "lang2": "description",
        "comparison": "which is easier"
    },
    "performance": {
        "lang1": "description",
        "lang2": "description",
        "winner": "better performing language"
    },
    "use_cases": {
        "lang1": ["use case1", "use case2"],
        "lang2": ["use case1", "use case2"]
    },
    "job_market": {
        "demand": "comparison of demand",
        "salary_range": "comparison of salaries",
        "growth_trend": "market trend analysis"
    }
}

Language 1: ${lang1}
Language 2: ${lang2}
""" 