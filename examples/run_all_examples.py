import asyncio
import os
from dotenv import load_dotenv

# Import all example modules
from basic_usage import main as basic_main
from basic_openai_example import main as openai_main
from basic_claude_example import main as claude_main
from advanced_usage import main as advanced_main
from custom_templates import main as custom_main
from comparison_provider import main as comparison_provider_main
from comparison_streaming_example import main as comparison_streaming_main
from comparison_provider_chaining import main as comparison_chaining_main

load_dotenv()

# Validate environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def run_example(name: str, func) -> None:
    """Run a single example with proper error handling"""
    try:
        print("\n" + "="*50)
        print(f"Running {name}")
        print("="*50)
        await func()
    except Exception as e:
        print(f"\nError in {name}: {str(e)}")

async def run_all_examples():
    """Run all example files"""
    examples = [
        ("Basic Usage", basic_main),
        ("OpenAI Examples", openai_main),
        ("Claude Examples", claude_main),
        ("Advanced Usage", advanced_main),
        ("Custom Templates", custom_main),
        ("Provider Comparison", comparison_provider_main),
        ("Streaming Examples", comparison_streaming_main),
        ("Provider Chaining", comparison_chaining_main)
    ]

    for name, func in examples:
        await run_example(name, func)

def main():
    """Main entry point"""
    try:
        asyncio.run(run_all_examples())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\n" + "="*50)
        print("Finished running all examples")
        print("="*50)

if __name__ == "__main__":
    main() 