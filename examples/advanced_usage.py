import asyncio
import os
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """Advanced usage examples demonstrating various LLMEasy features"""
    load_dotenv()
    
    llm = LLMEasy(
        provider='claude',
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    # Process multiple queries sequentially for now
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Java?"
    ]
    
    print("\nSequential Processing Results:")
    for prompt in prompts:
        response = await llm.query(
            prompt=prompt,
            system="You are a programming expert.",
            temperature=0.7
        )
        print(f"\nResponse for '{prompt}':\n{response}")
        
    # Streaming example
    print("\nStreaming Response:")
    async for chunk in llm.stream(
        prompt="Explain microservices architecture",
        system="You are a software architect."
    ):
        print(chunk, end="", flush=True)
        
    # JSON response example
    result = await llm.query(
        prompt="List 3 programming languages and their key features",
        system="You are a programming expert.",
        response_format="json"
    )
    print("\n\nJSON Response:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 