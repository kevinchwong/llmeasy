import asyncio
import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from llmeasy import LLMEasy

async def main():
    """Demonstrates different custom template examples with LLMEasy"""
    load_dotenv()
    
    llm = LLMEasy(
        provider='claude',
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        debug=True  # Enable debug mode
    )
    
    # Simple template for programming language description
    language_template = {
        "name": "string",
        "description": "string", 
        "features": ["string"]
    }
    
    # More complex template with nested structure
    framework_template = {
        "name": "string",
        "type": "string",
        "details": {
            "language": "string",
            "use_cases": ["string"],
            "learning_curve": "string"
        },
        "version": "string"
    }

    try:
        print("\nTemplate-based Programming Language Response:")
        print("Sending request with template:", json.dumps(language_template, indent=2))
        print("\nStarting stream_json process...")
        
        raw_response = ""
        count = 0
        async for json_obj in llm.stream_json(
            prompt="Describe Python programming language",
            system="You are a programming expert. Provide detailed technical information.",
            template=language_template
        ):
            count += 1
            print(f"\nReceived chunk #{count}:")
            print(f"Raw chunk data: {json_obj}")
            print(f"Type of response: {type(json_obj)}")
            
            # Accumulate raw response for debugging
            if isinstance(json_obj, str):
                raw_response += json_obj
            
            try:
                formatted_json = json.dumps(json_obj, indent=2)
                print(f"Formatted JSON:\n{formatted_json}")
            except Exception as json_err:
                print(f"Error formatting JSON: {str(json_err)}")
            
            print("-" * 50)

        print("\nFinal accumulated raw response:")
        print(raw_response)
        print("\nStream processing completed.")

        print("\nTemplate-based Framework Response:")
        print("Sending request with template:", json.dumps(framework_template, indent=2))
        print("\nStarting stream_json process...")
        
        raw_response = ""
        count = 0
        async for json_obj in llm.stream_json(
            prompt="Describe the Django web framework",
            system="You are a web development expert. Focus on practical aspects.",
            template=framework_template
        ):
            count += 1
            print(f"\nReceived chunk #{count}:")
            print(f"Raw chunk data: {json_obj}")
            print(f"Type of response: {type(json_obj)}")
            
            # Accumulate raw response for debugging
            if isinstance(json_obj, str):
                raw_response += json_obj
            
            try:
                formatted_json = json.dumps(json_obj, indent=2)
                print(f"Formatted JSON:\n{formatted_json}")
            except Exception as json_err:
                print(f"Error formatting JSON: {str(json_err)}")
            
            print("-" * 50)

        print("\nFinal accumulated raw response:")
        print(raw_response)
        print("\nStream processing completed.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 