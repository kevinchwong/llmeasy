import asyncio
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.basic_templates import STORY_TEMPLATE, SENTENCE_ANALYSIS_TEMPLATE

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

async def basic_example():
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "style": "humorous",
        "subject": "a talking coffee machine",
        "word_count": "100"
    }
    
    try:
        response = await llm.query(
            template=STORY_TEMPLATE,
            variables=variables
        )
        print("Story:", response)
    except Exception as e:
        print(f"Error in basic example: {e}")

async def json_example():
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "sentence": "The quick brown fox jumps over the lazy dog."
    }
    
    try:
        response = await llm.query(
            template=SENTENCE_ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print("\nSentence Analysis:", response)
    except Exception as e:
        print(f"Error in json example: {e}")

async def main():
    try:
        print("=== Basic Example ===")
        await basic_example()
        
        print("\n=== JSON Example ===")
        await json_example()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())  # Run both examples in a single event loop 