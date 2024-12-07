import asyncio
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.mistral_templates import (
    CODE_REVIEW_TEMPLATE,
    EXPLANATION_TEMPLATE,
    PROBLEM_SOLVING_TEMPLATE
)

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")

async def mistral_code_review():
    """Example of using Mistral for code review"""
    llm = LLMEasy(
        provider='mistral',
        api_key=MISTRAL_API_KEY,
        temperature=0.3
    )
    
    variables = {
        "code": """
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + middle + quicksort(right)
        """
    }
    
    try:
        response = await llm.query(
            template=CODE_REVIEW_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("=== Code Review Example ===")
        print(response)
    except Exception as e:
        print(f"Error in code review: {e}")

async def mistral_explanation():
    """Example of using Mistral for explanations"""
    llm = LLMEasy(
        provider='mistral',
        api_key=MISTRAL_API_KEY
    )
    
    variables = {
        "topic": "neural networks",
        "complexity": "intermediate",
        "focus": "practical applications"
    }
    
    try:
        response = await llm.query(
            template=EXPLANATION_TEMPLATE,
            variables=variables
        )
        print("\n=== Explanation Example ===")
        print(response)
    except Exception as e:
        print(f"Error in explanation: {e}")

async def mistral_problem_solving():
    """Example of using Mistral for problem solving"""
    llm = LLMEasy(
        provider='mistral',
        api_key=MISTRAL_API_KEY,
        temperature=0.7
    )
    
    variables = {
        "problem": """
        A company needs to optimize their delivery routes for 5 trucks
        delivering to 50 locations in a city. Each truck has a maximum
        capacity and working hours. How would you approach this problem?
        """,
        "constraints": """
        - Each truck can carry up to 1000kg
        - Working hours: 8am to 5pm
        - Some locations have time windows for delivery
        - Need to minimize fuel consumption
        """
    }
    
    try:
        response = await llm.query(
            template=PROBLEM_SOLVING_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("\n=== Problem Solving Example ===")
        print(response)
    except Exception as e:
        print(f"Error in problem solving: {e}")

async def main():
    try:
        await mistral_code_review()
        await mistral_explanation()
        await mistral_problem_solving()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 