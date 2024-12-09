import asyncio
import os
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """Basic usage example"""
    load_dotenv()
    
    llm = LLMEasy(
        provider='claude',
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    # Basic query example
    response = await llm.query(
        prompt="What is Python programming language?",
        system="You are a helpful programming expert."
    )
    print("\nBasic Query Response:")
    print(response)
    
    # Streaming example
    print("\nStreaming Response:")
    async for chunk in llm.stream(
        prompt="Explain what is an API in simple terms.",
        system="You are a helpful tech expert."
    ):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    asyncio.run(main())  # Run both examples in a single event loop 