import asyncio
from typing import Dict, Any, AsyncGenerator
from llmeasy import LLMEasy
from dotenv import load_dotenv
import os

# Import template or define fallback
try:
    from templates.comparison_templates import TECHNICAL_EXPLANATION_TEMPLATE
except ImportError:
    TECHNICAL_EXPLANATION_TEMPLATE = {
        "system": "You are an expert technical writer explaining complex topics clearly.",
        "prompt": "Explain {topic} to {audience}, focusing on practical applications and key concepts."
    }

def validate_api_keys() -> tuple[str, str]:
    """Validate that required API keys are present"""
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    missing_keys = []
    if not anthropic_key:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not openai_key:
        missing_keys.append("OPENAI_API_KEY")
        
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
    return anthropic_key, openai_key

async def stream_openai_response(
    llm: LLMEasy,
    variables: Dict[str, str]
) -> AsyncGenerator[str, None]:
    """Handle OpenAI streaming response"""
    print("\n=== OpenAI Streaming ===")
    print(f"Topic: {variables['topic']}")
    
    try:
        async for chunk in await llm.stream(
            template=TECHNICAL_EXPLANATION_TEMPLATE,
            variables=variables
        ):
            yield chunk
    except Exception as e:
        raise Exception(f"OpenAI streaming error: {str(e)}")

async def stream_claude_response(
    llm: LLMEasy,
    variables: Dict[str, str]
) -> AsyncGenerator[str, None]:
    """Handle Claude streaming response"""
    print("\n=== Claude Streaming ===")
    print(f"Topic: {variables['topic']}")
    
    try:
        async for chunk in await llm.stream(
            template=TECHNICAL_EXPLANATION_TEMPLATE,
            variables=variables
        ):
            yield chunk
    except Exception as e:
        raise Exception(f"Claude streaming error: {str(e)}")

async def main():
    """
    Example demonstrating streaming responses from different LLM providers.
    Compares streaming capabilities of OpenAI and Claude using the same template.
    """
    load_dotenv()
    
    try:
        # Validate API keys
        anthropic_key, openai_key = validate_api_keys()
        
        # Initialize providers
        openai = LLMEasy(provider='openai', api_key=openai_key)
        claude = LLMEasy(provider='claude', api_key=anthropic_key)

        # OpenAI streaming example
        variables = {
            "topic": "asynchronous programming in Python",
            "audience": "intermediate developers"
        }
        
        async for chunk in stream_openai_response(openai, variables):
            print(chunk, end='', flush=True)
        print()  # Add newline at the end

        # Claude streaming example
        variables = {
            "topic": "functional programming concepts",
            "audience": "intermediate developers"
        }
        
        async for chunk in stream_claude_response(claude, variables):
            print(chunk, end='', flush=True)
        print()  # Add newline at the end

    except Exception as e:
        print(f"Error in streaming example: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 