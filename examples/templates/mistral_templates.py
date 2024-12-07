CODE_REVIEW_TEMPLATE = """
Review the following code and provide feedback in JSON format:
{
    "complexity": "assessment of code complexity",
    "efficiency": "performance analysis",
    "best_practices": ["practice1", "practice2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "security_concerns": ["concern1", "concern2"]
}

Code:
${code}
"""

EXPLANATION_TEMPLATE = """
Provide a ${complexity}-level explanation of ${topic} with a focus on ${focus}.
"""

PROBLEM_SOLVING_TEMPLATE = """
Analyze the following problem and provide a solution approach in JSON format:
{
    "problem_breakdown": ["step1", "step2"],
    "key_considerations": ["consideration1", "consideration2"],
    "proposed_solution": {
        "approach": "solution description",
        "steps": ["step1", "step2"],
        "tools_needed": ["tool1", "tool2"]
    },
    "alternative_approaches": ["approach1", "approach2"],
    "potential_challenges": ["challenge1", "challenge2"]
}

Problem:
${problem}

Constraints:
${constraints}
""" 