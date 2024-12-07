import asyncio
import json
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os
from templates.comparison_templates import CODE_ANALYSIS_TEMPLATE

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def main():
    """Example comparing responses from different providers"""
    try:
        claude = LLMEasy(
            provider='claude',
            api_key=ANTHROPIC_API_KEY,
            temperature=0.7
        )
        
        openai = LLMEasy(
            provider='openai',
            api_key=OPENAI_API_KEY,
            model='gpt-4-turbo-preview',
            temperature=0.7
        )

        variables = {
            "code": """
            def find_duplicates(arr):
                seen = set()
                duplicates = []
                for num in arr:
                    if num in seen:
                        duplicates.append(num)
                    seen.add(num)
                return duplicates
            """
        }

        print("=== Claude Analysis ===")
        claude_response = await claude.query(
            template=CODE_ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print(json.dumps(claude_response, indent=2))

        print("\n=== OpenAI Analysis ===")
        openai_response = await openai.query(
            template=CODE_ANALYSIS_TEMPLATE,
            variables=variables,
            output_format='json'
        )
        print(json.dumps(openai_response, indent=2))
    except Exception as e:
        print(f"Error in compare_providers: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 