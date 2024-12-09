import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Any
from dotenv import load_dotenv
from llmeasy import LLMEasy
from llmeasy.utils.config import settings

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

async def stream_programming_languages() -> AsyncGenerator[Dict[Any, Any], None]:
    """Stream programming language information as JSON from Claude"""
    
    llm = LLMEasy(
        provider='claude',
        api_key=ANTHROPIC_API_KEY,
        temperature=0.3
    )
    
    # Define the JSON structure template
    json_template = {
        "name": "string",
        "year_created": "number",
        "creator": "string",
        "paradigm": ["string"],
        "use_cases": ["string"]
    }
    
    # Make the prompts more explicit about JSON formatting
    system_prompt = """You are a programming expert who always responds in valid JSON.
    Output exactly 5 programming languages, one at a time.
    Each response must be a single, complete, valid JSON object.
    Do not include any explanatory text or formatting.
    Example format:
    {"name": "Python", "year_created": 1991, "creator": "Guido van Rossum", "paradigm": ["object-oriented", "imperative"], "use_cases": ["web development", "data science"]}"""
    
    user_prompt = """Generate information about 5 major programming languages.
    For each language, provide a complete JSON object with these exact fields:
    - name (string)
    - year_created (number)
    - creator (string)
    - paradigm (array of strings)
    - use_cases (array of strings)
    
    Output one language at a time as a complete JSON object.
    Include only factual, accurate information."""
    
    print("Starting JSON stream...")  # Debug point
    
    try:
        async for language_data in llm.stream_json(
            prompt=user_prompt,
            system=system_prompt,
            template=json_template,
            repair_json=True
        ):
            print(f"Received language data: {language_data}")  # Debug point
            yield language_data
    except Exception as e:
        print(f"Error in stream_programming_languages: {str(e)}")
        raise

async def process_language_data(language_data: Dict[Any, Any]) -> None:
    """Process each language data object"""
    try:
        name = language_data.get("name", "Unknown")
        year = language_data.get("year_created", "N/A")
        creator = language_data.get("creator", "Unknown")
        
        print(f"\n📦 Processing: {name}")
        print(f"Created in {year} by {creator}")
        print("Details:", json.dumps(language_data, indent=2))
    except Exception as e:
        print(f"Error processing language data: {str(e)}")
        print(f"Raw data: {language_data}")

async def main():
    print("🚀 Starting programming language information stream...")
    count = 0
    
    try:
        async for language_data in stream_programming_languages():
            count += 1
            print(f"\n--- Processing language {count} ---")
            await process_language_data(language_data)
            
    except KeyboardInterrupt:
        print("\n⚠️ Stream interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    else:
        print(f"\n✅ Stream completed successfully. Processed {count} languages")

if __name__ == "__main__":
    asyncio.run(main()) 