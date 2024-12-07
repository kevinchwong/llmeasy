import asyncio
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.grok_templates import (
    ANALYSIS_TEMPLATE,
    CODING_TEMPLATE,
    DEBUGGING_TEMPLATE,
    EXPLANATION_TEMPLATE
)

load_dotenv()

GROK_API_KEY = os.getenv('GROK_API_KEY')
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY environment variable is not set")

async def grok_analysis_example():
    """Example of using Grok for analysis"""
    llm = LLMEasy(
        provider='grok',
        api_key=GROK_API_KEY,
        temperature=0.3
    )
    
    variables = {
        "code": """
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
                
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            
            return merge(left, right)
            
        def merge(left, right):
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
                    
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        """
    }
    
    try:
        response = await llm.query(
            template=ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("=== Code Analysis Example ===")
        print(response)
    except Exception as e:
        print(f"Error in analysis example: {e}")

async def grok_coding_example():
    """Example of using Grok for code generation"""
    llm = LLMEasy(
        provider='grok',
        api_key=GROK_API_KEY,
        temperature=0.7
    )
    
    variables = {
        "task": "Create a function that implements a binary search tree with insert and search operations",
        "language": "Python",
        "style": "object-oriented"
    }
    
    try:
        response = await llm.query(
            template=CODING_TEMPLATE,
            variables=variables
        )
        print("\n=== Code Generation Example ===")
        print(response)
    except Exception as e:
        print(f"Error in coding example: {e}")

async def grok_debugging_example():
    """Example of using Grok for debugging"""
    llm = LLMEasy(
        provider='grok',
        api_key=GROK_API_KEY
    )
    
    variables = {
        "code": """
        def fibonacci(n):
            if n <= 0:
                return []
            elif n == 1:
                return [0]
                
            sequence = [0, 1]
            while len(sequence) < n:
                sequence.append(sequence[-1] + sequence[-2])
                
            return sequence
        
        # Test cases that fail
        print(fibonacci(0))  # Should return []
        print(fibonacci(1))  # Should return [0]
        print(fibonacci(5))  # Should return [0, 1, 1, 2, 3]
        """,
        "error": "The function doesn't handle edge cases correctly",
        "expected_behavior": "Should return correct Fibonacci sequences for all inputs including edge cases"
    }
    
    try:
        response = await llm.query(
            template=DEBUGGING_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("\n=== Debugging Example ===")
        print(response)
    except Exception as e:
        print(f"Error in debugging example: {e}")

async def grok_explanation_example():
    """Example of using Grok for technical explanations"""
    llm = LLMEasy(
        provider='grok',
        api_key=GROK_API_KEY
    )
    
    variables = {
        "concept": "blockchain technology",
        "audience": "software developers",
        "focus": "technical implementation details",
        "code_examples": "Include code examples to illustrate key points."
    }
    
    try:
        response = await llm.query(
            template=EXPLANATION_TEMPLATE,
            variables=variables
        )
        print("\n=== Technical Explanation Example ===")
        print(response)
    except Exception as e:
        print(f"Error in explanation example: {e}")

async def main():
    try:
        await grok_analysis_example()
        await grok_coding_example()
        await grok_debugging_example()
        await grok_explanation_example()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 