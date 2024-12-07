# Templates for claude_example.py

BASIC_EXPLANATION_TEMPLATE = """
Write a ${tone} explanation of ${concept} 
that a ${audience} would understand.
Keep it under ${length} words.
"""

CODE_ANALYSIS_TEMPLATE = """
Analyze this code snippet and provide a detailed analysis in JSON format with the following structure:
{
    "time_complexity": "explanation",
    "space_complexity": "explanation",
    "optimizations": ["suggestion1", "suggestion2"],
    "edge_cases": ["case1", "case2"],
    "best_practices": ["practice1", "practice2"]
}

Code:
def find_pairs(arr, target_sum):
    seen = set()
    pairs = []
    for num in arr:
        complement = target_sum - num
        if complement in seen:
            pairs.append((num, complement))
        seen.add(num)
    return pairs
"""

CREATIVE_WRITING_TEMPLATE = """
Write a ${genre} story that includes these elements:
- Setting: ${setting}
- Main character: ${character}
- Theme: ${theme}
- Must include this phrase: "${phrase}"
- Length: Exactly ${length} words

Make it engaging and memorable.
"""

DATA_EXTRACTION_TEMPLATE = """
Extract and structure the following information from this research paper abstract in this exact JSON format:
{
    "research_question": {
        "main_focus": "primary research focus",
        "target_population": "study population"
    },
    "methodology": {
        "approach": "detailed description of method",
        "sample_size": {
            "number": "numeric value",
            "unit": "participants/subjects"
        },
        "duration": {
            "length": "numeric value",
            "unit": "time unit"
        },
        "measurements": ["measurement1", "measurement2"]
    },
    "key_findings": [
        {
            "finding": "description",
            "significance": "statistical or practical significance"
        }
    ],
    "limitations": [
        {
            "limitation": "description",
            "impact": "effect on results"
        }
    ],
    "future_work": [
        {
            "suggestion": "description",
            "rationale": "why this is important"
        }
    ]
}

Abstract:
${abstract}
""" 