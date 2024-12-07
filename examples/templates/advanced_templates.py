# Templates for advanced_usage.py

CODE_GENERATION_TEMPLATE = """
Write a Python function that ${task}.

Requirements:
- Include type hints
- Include docstring
- Follow PEP 8
- Include error handling

Function name: ${function_name}
"""

DATA_ANALYSIS_TEMPLATE = """
Analyze the following sales data and provide the analysis in this JSON format:
{
    "total_revenue": {
        "amount": "total dollar amount",
        "currency": "USD"
    },
    "best_performing_product": {
        "name": "product name",
        "revenue": "total revenue for this product",
        "growth_rate": "percentage growth"
    },
    "growth_trend": {
        "overall": "trend description",
        "by_product": {
            "Product A": "trend description",
            "Product B": "trend description",
            "Product C": "trend description"
        }
    },
    "recommendations": [
        "recommendation1",
        "recommendation2",
        "recommendation3"
    ]
}

Sales Data:
${sales_data}
"""

CHAIN_OF_THOUGHT_TEMPLATE = """
Let's solve this problem step by step:

Problem: ${problem}

Please:
1. Break down the problem
2. Show your reasoning for each step
3. Provide the final answer
""" 