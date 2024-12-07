import asyncio
import json
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.claude_templates import (
    BASIC_EXPLANATION_TEMPLATE,
    CODE_ANALYSIS_TEMPLATE,
    CREATIVE_WRITING_TEMPLATE,
    DATA_EXTRACTION_TEMPLATE
)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

async def claude_basic_example():
    """Basic text generation with Claude"""
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "tone": "friendly and engaging",
        "concept": "quantum entanglement",
        "audience": "high school student",
        "length": "150"
    }
    
    try:
        response = await llm.query(
            template=BASIC_EXPLANATION_TEMPLATE,
            variables=variables
        )
        print("=== Basic Example ===")
        print(response)
    except Exception as e:
        print(f"Error in basic example: {e}")

async def claude_structured_analysis():
    """Example of getting structured JSON analysis"""
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
        temperature=0.3
    )
    
    try:
        response = await llm.query(
            template=CODE_ANALYSIS_TEMPLATE,
            variables={},
            output_format='json'
        )
        print("\n=== Code Analysis ===")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error in structured analysis: {e}")

async def claude_creative_writing():
    """Example of creative writing with specific constraints"""
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
        temperature=0.9
    )
    
    variables = {
        "genre": "science fiction",
        "setting": "a space station orbiting a black hole",
        "character": "a quantum physicist with a secret",
        "theme": "the nature of time",
        "phrase": "the clock struck thirteen",
        "length": "200"
    }
    
    try:
        response = await llm.query(
            template=CREATIVE_WRITING_TEMPLATE,
            variables=variables
        )
        print("\n=== Creative Writing ===")
        print(response)
    except Exception as e:
        print(f"Error in creative writing: {e}")

async def claude_data_extraction():
    """Example of extracting structured data from text"""
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "abstract": """
        This study investigated the impact of mindfulness meditation on stress levels
        among college students during exam periods. Using a randomized controlled trial
        with 120 participants, we measured cortisol levels and self-reported stress
        before and after a 6-week meditation program. Results showed a significant
        reduction in both physiological and perceived stress in the meditation group
        compared to controls. However, the small sample size and single-institution
        focus limit generalizability. Future research should examine long-term effects
        and include multiple universities.
        """
    }
    
    try:
        response = await llm.query(
            template=DATA_EXTRACTION_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("\n=== Data Extraction ===")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error in data extraction: {e}")

async def main():
    try:
        await claude_basic_example()
        await claude_structured_analysis()
        await claude_creative_writing()
        await claude_data_extraction()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 