import asyncio
import os
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """Provider comparison example"""
    load_dotenv()
    
    providers = ['claude', 'openai']
    responses = {}
    
    for provider in providers:
        llm = LLMEasy(
            provider=provider,
            api_key=os.getenv(f'{provider.upper()}_API_KEY')
        )
        
        response = await llm.query(
            prompt="What is artificial intelligence?",
            system="You are an AI expert."
        )
        responses[provider] = response
        
    for provider, response in responses.items():
        print(f"\n{provider.upper()} Response:")
        print(response)

if __name__ == "__main__":
    asyncio.run(main()) 