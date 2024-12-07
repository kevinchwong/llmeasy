import asyncio
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.openai_templates import (
    POEM_TEMPLATE,
    REVIEW_ANALYSIS_TEMPLATE,
    LANGUAGE_COMPARISON_TEMPLATE
)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def openai_basic_example():
    llm = LLMEasy(
        provider='openai',
        api_key=OPENAI_API_KEY,
        model='gpt-4-turbo-preview'
    )
    
    variables = {
        "length": "short",
        "style": "haiku",
        "subject": "artificial intelligence"
    }
    
    try:
        response = await llm.query(
            template=POEM_TEMPLATE,
            variables=variables
        )
        print("Poem:", response)
    except Exception as e:
        print(f"Error in basic example: {e}")

async def openai_json_example():
    llm = LLMEasy(
        provider='openai',
        api_key=OPENAI_API_KEY
    )
    
    variables = {
        "review": """
        The latest superhero movie was a mixed bag. While the special effects 
        were stunning and the action sequences kept me on the edge of my seat,
        the plot felt somewhat predictable. The lead actor gave a strong 
        performance, but some of the supporting characters were underdeveloped.
        Still, it's worth watching for the visuals alone.
        """
    }
    
    try:
        response = await llm.query(
            template=REVIEW_ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("\nReview Analysis:", response)
    except Exception as e:
        print(f"Error in json example: {e}")

async def openai_comparison_example():
    llm = LLMEasy(
        provider='openai',
        api_key=OPENAI_API_KEY,
        temperature=0.3
    )
    
    variables = {
        "lang1": "Python",
        "lang2": "Rust"
    }
    
    try:
        response = await llm.query(
            template=LANGUAGE_COMPARISON_TEMPLATE,
            variables=variables,
            output_format='json',
            max_tokens=1000
        )
        print("\nLanguage Comparison:", response)
    except Exception as e:
        print(f"Error in comparison example: {e}")

async def main():
    try:
        print("=== Basic OpenAI Example ===")
        await openai_basic_example()
        
        print("\n=== JSON Output Example ===")
        await openai_json_example()
        
        print("\n=== Comparison Example ===")
        await openai_comparison_example()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 