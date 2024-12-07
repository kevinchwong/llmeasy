import asyncio
import os
from llm_query import LLMQuery
from dotenv import load_dotenv
from templates.advanced_templates import (
    CODE_GENERATION_TEMPLATE,
    DATA_ANALYSIS_TEMPLATE,
    CHAIN_OF_THOUGHT_TEMPLATE
)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

async def code_generation_example():
    """Example of generating code with specific formatting"""
    llm = LLMQuery(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1
    )
    
    variables = {
        "task": "validates if a string is a valid email address",
        "function_name": "validate_email"
    }
    
    response = await llm.query(
        template=CODE_GENERATION_TEMPLATE,
        variables=variables,
        output_format='python'
    )
    
    print("Generated Code:", response)

async def data_analysis_example():
    """Example of analyzing data with structured output"""
    llm = LLMQuery(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
    )
    
    variables = {
        "sales_data": """
        Product A: $1200 (Q1), $1500 (Q2), $1800 (Q3)
        Product B: $800 (Q1), $1000 (Q2), $950 (Q3)
        Product C: $2000 (Q1), $1800 (Q2), $2200 (Q3)
        """
    }
    
    response = await llm.query(
        template=DATA_ANALYSIS_TEMPLATE,
        variables=variables,
        output_format='json',
        max_tokens=2000
    )
    
    print("Analysis Results:", response)

async def chain_of_thought_example():
    """Example of using chain-of-thought prompting"""
    llm = LLMQuery(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
    )
    
    variables = {
        "problem": "If a train travels 120 km in 2 hours with a 15-minute stop, what was its actual average speed while moving?"
    }
    
    response = await llm.query(
        template=CHAIN_OF_THOUGHT_TEMPLATE,
        variables=variables
    )
    
    print("Step-by-step Solution:", response)

async def main():
    print("=== Code Generation Example ===")
    await code_generation_example()
    
    print("\n=== Data Analysis Example ===")
    await data_analysis_example()
    
    print("\n=== Chain of Thought Example ===")
    await chain_of_thought_example()

if __name__ == "__main__":
    asyncio.run(main()) 