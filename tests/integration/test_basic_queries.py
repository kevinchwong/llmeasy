import pytest
from llmeasy import LLMEasy
import asyncio
from tests.integration.test_utils import log_test_failure, cleanup_provider
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_query(http_client, api_keys):
    """Test basic query functionality across providers"""
    providers = {
        'claude': {
            'api_key': api_keys['ANTHROPIC_API_KEY'],
            'expected_reliability': True,
            'known_issues': [],
            'timeout': 30,
            'retry_attempts': 2
        },
        'openai': {
            'api_key': api_keys['OPENAI_API_KEY'],
            'expected_reliability': True,
            'known_issues': [],
            'timeout': 30,
            'retry_attempts': 2
        },
        'mistral': {
            'api_key': api_keys['MISTRAL_API_KEY'],
            'expected_reliability': True,
            'known_issues': ['rate limiting', 'connection stability'],
            'timeout': 40,
            'retry_attempts': 3
        },
        'gemini': {
            'api_key': api_keys['GOOGLE_API_KEY'],
            'expected_reliability': True,
            'known_issues': ['response format inconsistency'],
            'timeout': 35,
            'retry_attempts': 2
        },
        'grok': {
            'api_key': api_keys['GROK_API_KEY'],
            'expected_reliability': True,
            'known_issues': ['beta API', 'rate limiting'],
            'timeout': 40,
            'retry_attempts': 3
        }
    }
    
    for provider_name, config in providers.items():
        if not config['api_key']:
            logger.info(f"Skipping {provider_name} - No API key provided")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing provider: {provider_name}")
        logger.info(f"{'='*50}")
        
        retry_count = 0
        while retry_count <= config['retry_attempts']:
            try:
                llm = LLMEasy(
                    provider=provider_name, 
                    api_key=config['api_key'],
                    temperature=0.7
                )
                
                async with asyncio.timeout(config['timeout']):
                    logger.info(f"Sending query to {provider_name}...")
                    try:
                        response = await llm.query(
                            prompt="What is Python? Provide a brief explanation.",
                            system="You are a programming expert. Be concise."
                        )
                        
                        # Validate response
                        assert isinstance(response, str), f"Invalid response type: {type(response)}"
                        assert len(response) > 0, "Empty response received"
                        assert len(response.split()) >= 5, "Response too short"
                        
                        logger.info(f"✅ Successfully tested {provider_name}")
                        logger.info(f"Response preview: {response[:100]}...")
                        break  # Success, exit retry loop
                        
                    except AttributeError as e:
                        if 'chat_async' in str(e):
                            log_test_failure(
                                provider_name,
                                e,
                                context="Provider does not support async chat method."
                            )
                            raise
                        raise
                    
            except Exception as e:
                retry_count += 1
                log_test_failure(
                    provider_name, 
                    e,
                    context=f"Attempt {retry_count} of {config['retry_attempts']+1}"
                )
                
                if retry_count <= config['retry_attempts']:
                    delay = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"Retrying {provider_name} in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    if config['expected_reliability']:
                        logger.error(f"All retry attempts failed for {provider_name}")
                        pytest.fail(f"Unexpected failure for reliable provider {provider_name}")
                    else:
                        logger.warning(f"Known issues for {provider_name}:")
                        for issue in config['known_issues']:
                            logger.warning(f"  - {issue}")
                
            finally:
                if 'llm' in locals():
                    try:
                        await cleanup_provider(llm)
                    except Exception as e:
                        logger.error(f"Error during cleanup for {provider_name}: {e}")
                    await asyncio.sleep(1)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_system_message_handling(api_keys):
    """Test how providers handle system messages"""
    providers = ['claude', 'openai', 'mistral']
    system_messages = [
        "You are a helpful assistant",
        "Respond in a technical manner",
        "Keep responses under 50 words",
        None  # Test with no system message
    ]
    
    for provider in providers:
        if not api_keys.get(f'{provider.upper()}_API_KEY'):
            continue
            
        llm = None
        try:
            llm = LLMEasy(
                provider=provider,
                api_key=api_keys[f'{provider.upper()}_API_KEY']
            )
            
            for system in system_messages:
                response = await llm.query(
                    prompt="What is Python?",
                    system=system
                )
                
                assert isinstance(response, str)
                assert len(response) > 0
                
                if system and "under 50 words" in system:
                    assert len(response.split()) <= 50, "System message word limit not respected"
                    
        finally:
            if llm:
                await cleanup_provider(llm)