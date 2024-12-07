ANALYSIS_TEMPLATE = """
Analyze the following code and provide insights in JSON format:
{
    "complexity": {
        "time": "time complexity analysis",
        "space": "space complexity analysis",
        "cognitive": "cognitive complexity assessment"
    },
    "best_practices": {
        "followed": ["practice1", "practice2"],
        "suggestions": ["suggestion1", "suggestion2"]
    },
    "optimization": {
        "opportunities": ["opportunity1", "opportunity2"],
        "trade_offs": ["trade_off1", "trade_off2"]
    },
    "security": {
        "concerns": ["concern1", "concern2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }
}

Code:
${code}
"""

CODING_TEMPLATE = """
Write ${language} code that ${task}. Use ${style} programming style.
Include comments and docstrings.
"""

DEBUGGING_TEMPLATE = """
Debug the following code and provide analysis in JSON format:
{
    "issues": {
        "identified": ["issue1", "issue2"],
        "root_causes": ["cause1", "cause2"]
    },
    "fixes": {
        "required_changes": ["change1", "change2"],
        "code_snippets": {
            "original": "problematic code",
            "fixed": "corrected code"
        }
    },
    "testing": {
        "edge_cases": ["case1", "case2"],
        "validation_steps": ["step1", "step2"]
    }
}

Code with error:
${code}

Error description:
${error}

Expected behavior:
${expected_behavior}
"""

EXPLANATION_TEMPLATE = """
Explain ${concept} to ${audience} with a focus on ${focus}.
${code_examples}
""" 