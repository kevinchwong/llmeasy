import asyncio
import os
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """OpenAI-specific example"""
    load_dotenv()
    
    llm = LLMEasy(
        provider='openai',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    response = await llm.query(
        prompt="What are the key features of Python?",
        system="You are a Python expert."
    )
    print("\nOpenAI Response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main()) 