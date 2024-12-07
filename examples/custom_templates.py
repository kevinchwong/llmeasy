import asyncio
from llmeasy import LLMEasy
from llmeasy.templates import PromptTemplate
from dotenv import load_dotenv
import os
from templates.custom_templates import ANALYSIS_TEMPLATE, COMPARISON_TEMPLATE

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

async def literary_analysis():
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "content_type": "poem",
        "content": "Two roads diverged in a yellow wood...",
        "aspect1": "symbolism",
        "aspect2": "theme",
        "aspect3": "mood"
    }
    
    response = await llm.query(
        template=ANALYSIS_TEMPLATE,
        variables=variables,
        output_format='json'
    )
    
    print("Literary Analysis:", response)

async def product_comparison():
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY
    )
    
    variables = {
        "item1": "iPhone 15 Pro",
        "item2": "Samsung Galaxy S24 Ultra",
        "aspect1": "camera capabilities",
        "aspect2": "battery life",
        "aspect3": "user experience"
    }
    
    response = await llm.query(
        template=COMPARISON_TEMPLATE,
        variables=variables,
        output_format='json'
    )
    
    print("Product Comparison:", response)

async def main():
    try:
        print("=== Literary Analysis ===")
        await literary_analysis()
        
        print("\n=== Product Comparison ===")
        await product_comparison()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 