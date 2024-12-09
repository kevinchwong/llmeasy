import pytest
from llmeasy import LLMEasy
import asyncio
import json
from tests.integration.test_utils import retry_on_exception, log_test_failure, cleanup_provider
import logging

logger = logging.getLogger(__name__)

def validate_json_response(response: dict, provider: str) -> None:
    """Validate JSON response structure and content"""
    assert isinstance(response, dict), f"Response from {provider} is not a dictionary"
    
    def check_values(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                assert v is not None, f"Null value found at {new_path} from {provider}"
                if isinstance(v, str):
                    assert v.strip(), f"Empty string found at {new_path} from {provider}"
                check_values(v, new_path)
        elif isinstance(obj, list):
            assert obj, f"Empty list found at {path} from {provider}"
            for i, item in enumerate(obj):
                check_values(item, f"{path}[{i}]")
                
    check_values(response)

@retry_on_exception(retries=3)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_json_template_all_providers(api_keys):
    """Test JSON template functionality across providers"""
    providers = {
        'claude': {
            'api_key': api_keys['ANTHROPIC_API_KEY'],
            'expected_reliability': True,
            'timeout': 30
        },
        'openai': {
            'api_key': api_keys['OPENAI_API_KEY'],
            'expected_reliability': True,
            'timeout': 30
        }
    }
    
    # Alternative field names that providers might use
    field_mappings = {
        "name": ["name", "language_name", "title", "language"],
        "description": ["description", "overview", "summary", "about"],
        "features": ["features", "characteristics", "capabilities", "key_features", 
                    "key_points", "main_features", "core_features", "highlights"]
    }
    
    for provider_name, config in providers.items():
        if not config['api_key']:
            continue
            
        logger.info(f"\nTesting JSON template with {provider_name}")
        llm = None
        try:
            llm = LLMEasy(
                provider=provider_name, 
                api_key=config['api_key'],
                temperature=0.7
            )
            
            # Format the prompt to request specific JSON structure
            prompt = (
                "Describe Python programming language in JSON format with the following structure:\n"
                "{\n"
                '  "name": "string (name of the language)",\n'
                '  "description": "string (brief overview)",\n'
                '  "features": ["string (key features as an array)"]\n'
                "}"
            )
            
            async with asyncio.timeout(config['timeout']):
                response = await llm.query(
                    prompt=prompt,
                    system=(
                        "You are a programming expert. "
                        "Provide a structured response about Python. "
                        "Return only valid JSON without any additional text or markdown formatting."
                    ),
                    output_format='json'
                )
            
            logger.info(f"\nResponse from {provider_name}: {json.dumps(response, indent=2)}")
            
            validate_json_response(response, provider_name)
            
            # Flatten nested response for checking
            flat_response = {}
            def flatten_dict(d, parent_key=''):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten_dict(v, f"{parent_key}{k}_")
                    else:
                        flat_response[f"{parent_key}{k}"] = v
            
            flatten_dict(response)
            logger.debug(f"\nFlattened response: {json.dumps(flat_response, indent=2)}")
            
            # Check each required field using alternative names
            for field, alternatives in field_mappings.items():
                field_present = False
                used_field = None
                
                # Check direct matches
                for alt in alternatives:
                    if alt in flat_response:
                        field_present = True
                        used_field = alt
                        break
                
                # If not found, check for partial matches
                if not field_present:
                    for key in flat_response.keys():
                        if any(alt.lower() in key.lower() for alt in alternatives):
                            field_present = True
                            used_field = key
                            break
                
                assert field_present, (
                    f"No alternative for '{field}' found in {provider_name} response. "
                    f"Expected one of {alternatives}"
                )
                
                # Validate field content
                value = flat_response[used_field]
                if field == "features":
                    if isinstance(value, str):
                        # Convert comma-separated string to list if needed
                        value = [v.strip() for v in value.split(',')]
                    assert isinstance(value, list), f"Field '{used_field}' should be a list"
                    assert len(value) > 0, f"Field '{used_field}' is empty"
                else:
                    assert isinstance(value, str), f"Field '{used_field}' should be a string"
                    assert len(value) > 0, f"Field '{used_field}' is empty"
            
            logger.info(f"✅ Successfully tested JSON template with {provider_name}")
            
        except Exception as e:
            log_test_failure(
                provider_name,
                e,
                context=f"During JSON template test with config: {config}"
            )
            raise
            
        finally:
            if llm:
                await cleanup_provider(llm)
                await asyncio.sleep(1)