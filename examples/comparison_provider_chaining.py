import asyncio
import json
from llm_query import LLMQuery
from dotenv import load_dotenv
import os
from templates.comparison_templates import SUMMARY_TEMPLATE, EXPANSION_TEMPLATE

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def main():
    """Example showing how to chain different providers"""
    try:
        claude = LLMQuery(provider='claude', api_key=ANTHROPIC_API_KEY)
        openai = LLMQuery(provider='openai', api_key=OPENAI_API_KEY)

        variables = {
            "text": """
            Python's async/await syntax provides a way to write concurrent code that's 
            both efficient and readable. It allows you to write asynchronous code 
            that looks similar to synchronous code, making it easier to reason about 
            and maintain. The key components are coroutines (defined with async def), 
            the await keyword for waiting on coroutines, and event loops that manage 
            the execution of coroutines.
            """
        }

        # Get summary from Claude
        summary = await claude.query(
            template=SUMMARY_TEMPLATE,
            variables=variables,
            output_format='json'
        )

        # Use OpenAI to expand on the summary
        expansion = await openai.query(
            template=EXPANSION_TEMPLATE,
            variables={"points": json.dumps(summary)},
            output_format='json'
        )

        print("=== Original Summary (Claude) ===")
        print(json.dumps(summary, indent=2))
        print("\n=== Expanded Points (OpenAI) ===")
        print(json.dumps(expansion, indent=2))
    except Exception as e:
        print(f"Error in chain_providers: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 