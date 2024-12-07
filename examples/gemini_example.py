import asyncio
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.gemini_templates import (
    ANALYSIS_TEMPLATE,
    CREATIVE_TEMPLATE,
    TECHNICAL_TEMPLATE
)

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

async def gemini_analysis_example():
    """Example of using Gemini for analysis"""
    llm = LLMEasy(
        provider='gemini',
        api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    variables = {
        "text": """
        The rise of artificial intelligence has sparked both excitement and concern.
        While AI promises to revolutionize industries and solve complex problems,
        it also raises questions about job displacement and ethical implications.
        Recent advances in large language models and robotics have accelerated
        these discussions.
        """
    }
    
    try:
        response = await llm.query(
            template=ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("=== Analysis Example ===")
        print(response)
    except Exception as e:
        print(f"Error in analysis example: {e}")

async def gemini_creative_example():
    """Example of using Gemini for creative tasks"""
    llm = LLMEasy(
        provider='gemini',
        api_key=GOOGLE_API_KEY,
        temperature=0.9
    )
    
    variables = {
        "topic": "a day in the life of a quantum computer",
        "style": "whimsical",
        "length": "150 words"
    }
    
    try:
        response = await llm.query(
            template=CREATIVE_TEMPLATE,
            variables=variables
        )
        print("\n=== Creative Example ===")
        print(response)
    except Exception as e:
        print(f"Error in creative example: {e}")

async def gemini_technical_example():
    """Example of using Gemini for technical content"""
    llm = LLMEasy(
        provider='gemini',
        api_key=GOOGLE_API_KEY
    )
    
    variables = {
        "concept": "quantum entanglement",
        "audience": "high school students",
        "format": "step-by-step explanation"
    }
    
    try:
        response = await llm.query(
            template=TECHNICAL_TEMPLATE,
            variables=variables
        )
        print("\n=== Technical Example ===")
        print(response)
    except Exception as e:
        print(f"Error in technical example: {e}")

async def main():
    try:
        await gemini_analysis_example()
        await gemini_creative_example()
        await gemini_technical_example()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 