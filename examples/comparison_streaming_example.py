import asyncio
from llm_query import LLMQuery
from dotenv import load_dotenv
import os
from templates.comparison_templates import TECHNICAL_EXPLANATION_TEMPLATE

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def main():
    """Example showing how to use streaming responses"""
    try:
        # OpenAI streaming example
        print("\n=== OpenAI Streaming ===")
        print("Topic: Asynchronous Programming in Python")
        
        openai = LLMQuery(
            provider='openai',
            api_key=OPENAI_API_KEY
        )

        variables = {
            "topic": "asynchronous programming in Python",
            "audience": "intermediate developers"
        }

        async for chunk in await openai.stream(
            template=TECHNICAL_EXPLANATION_TEMPLATE,
            variables=variables
        ):
            print(chunk, end='', flush=True)
        print()  # Add newline at the end

        # Claude streaming example
        print("\n=== Claude Streaming ===")
        print("Topic: Functional Programming Concepts")
        
        claude = LLMQuery(
            provider='claude',
            api_key=ANTHROPIC_API_KEY
        )

        variables = {
            "topic": "functional programming concepts",
            "audience": "intermediate developers"
        }

        async for chunk in await claude.stream(
            template=TECHNICAL_EXPLANATION_TEMPLATE,
            variables=variables
        ):
            print(chunk, end='', flush=True)
        print()  # Add newline at the end

    except Exception as e:
        print(f"Error in streaming example: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 