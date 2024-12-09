import pytest
from llmeasy import LLMEasy
import asyncio
from integration.test_utils import log_test_failure, cleanup_provider
import logging
import json

logger = logging.getLogger(__name__)

def clean_json_response(response: str) -> dict:
    """Clean and parse JSON response"""
    try:
        if isinstance(response, dict):
            return response
        # Remove markdown formatting if present
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned.replace('```json', '', 1)
        if cleaned.endswith('```'):
            cleaned = cleaned.rsplit('```', 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {str(e)}\nResponse: {response}")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_specific_features(api_keys):
    """Test provider-specific features and configurations"""
    providers = {
        'claude': {
            'api_key': api_keys['ANTHROPIC_API_KEY'],
            'features': ['system_messages', 'streaming', 'json_mode']
        },
        'openai': {
            'api_key': api_keys['OPENAI_API_KEY'],
            'features': ['function_calling', 'streaming', 'json_mode']
        },
        'mistral': {
            'api_key': api_keys['MISTRAL_API_KEY'],
            'features': ['streaming', 'json_mode']
        }
    }
    
    for provider_name, config in providers.items():
        if not config['api_key']:
            continue
            
        llm = None
        try:
            llm = LLMEasy(
                provider=provider_name,
                api_key=config['api_key']
            )
            
            # Test each supported feature
            for feature in config['features']:
                logger.info(f"Testing {feature} for {provider_name}")
                
                if feature == 'system_messages':
                    response = await llm.query(
                        prompt="Hello",
                        system="Respond in French"
                    )
                    assert isinstance(response, str)
                    assert any(word in response.lower() for word in ['bonjour', 'salut'])
                    
                elif feature == 'json_mode':
                    raw_response = await llm.query(
                        prompt=(
                            "List two colors in JSON format with this structure:\n"
                            "{\n"
                            '  "colors": ["string", "string"]\n'
                            "}"
                        ),
                        system="Return only valid JSON",
                        output_format='json'
                    )
                    
                    # Clean and parse JSON response
                    response = clean_json_response(raw_response)
                    
                    assert isinstance(response, dict), f"Response is not a dictionary: {raw_response}"
                    assert 'colors' in response, f"Missing 'colors' key in response: {response}"
                    assert isinstance(response['colors'], list), f"'colors' is not a list: {response}"
                    assert len(response['colors']) == 2, f"Expected 2 colors, got: {response}"
                    
                elif feature == 'function_calling':
                    # Skip function calling test for now
                    logger.info(f"Skipping function calling test for {provider_name}")
                    continue
                    
                elif feature == 'streaming':
                    chunks = []
                    async for chunk in llm.stream(
                        prompt="Count to 3",
                        system="Be brief"
                    ):
                        assert isinstance(chunk, str)
                        chunks.append(chunk)
                    
                    complete_response = ''.join(chunks)
                    assert len(complete_response) > 0
                
                logger.info(f"✅ Successfully tested {feature} for {provider_name}")
                
        except Exception as e:
            log_test_failure(
                provider_name,
                e,
                context=f"Testing feature: {feature}"
            )
            if feature == 'json_mode':
                logger.error(f"Raw JSON response: {raw_response}")
            raise
            
        finally:
            if llm:
                await cleanup_provider(llm)
                await asyncio.sleep(1)