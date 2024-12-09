import asyncio
import os
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """Claude-specific example"""
    load_dotenv()
    
    llm = LLMEasy(
        provider='claude',
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        temperature=0.7
    )
    
    # JSON response example
    json_response = await llm.query(
        prompt="Describe Python programming language",
        system="You are a programming expert.",
        output_format='json'
    )
    print("\nJSON Response:")
    print(json_response)

if __name__ == "__main__":
    asyncio.run(main()) 