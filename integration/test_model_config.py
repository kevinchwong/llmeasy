import pytest
from llmeasy import LLMEasy
from integration.test_utils import log_test_failure, cleanup_provider
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_configuration(api_keys):
    """Test different model configurations"""
    model_configs = {
        'claude': {
            'api_key': api_keys['ANTHROPIC_API_KEY'],
            'models': ['claude-3-sonnet-20240229', 'claude-3-opus-20240229']
        },
        'openai': {
            'api_key': api_keys['OPENAI_API_KEY'],
            'models': ['gpt-4-turbo-preview', 'gpt-3.5-turbo']
        },
        'mistral': {
            'api_key': api_keys['MISTRAL_API_KEY'],
            'models': ['mistral-medium', 'mistral-small']
        }
    }
    
    for provider_name, config in model_configs.items():
        if not config['api_key']:
            continue
            
        for model in config['models']:
            llm = None
            try:
                logger.info(f"\nTesting {provider_name} with model: {model}")
                llm = LLMEasy(
                    provider=provider_name,
                    api_key=config['api_key'],
                    model=model
                )
                
                response = await llm.query(
                    prompt="What is Python?",
                    system="Be very brief."
                )
                
                assert isinstance(response, str)
                assert len(response) > 0
                
                logger.info(f"✅ Successfully tested {provider_name} with {model}")
                
            except Exception as e:
                log_test_failure(
                    provider_name,
                    e,
                    context=f"Testing model: {model}"
                )
                raise
            finally:
                if llm:
                    await cleanup_provider(llm) 

@pytest.mark.integration
@pytest.mark.asyncio
async def test_temperature_settings(api_keys):
    """Test different temperature settings affect response variety"""
    provider = 'openai'  # Use OpenAI as it's most reliable for this test
    temperatures = [0.0, 0.5, 1.0]
    responses = {}
    
    llm = None
    try:
        for temp in temperatures:
            llm = LLMEasy(
                provider=provider,
                api_key=api_keys['OPENAI_API_KEY'],
                temperature=temp
            )
            
            # Get multiple responses for each temperature
            temp_responses = []
            for _ in range(3):
                response = await llm.query(
                    prompt="What is Python?",
                    system="Be very brief."
                )
                temp_responses.append(response)
                await asyncio.sleep(1)  # Avoid rate limits
                
            responses[temp] = temp_responses
            
            # Cleanup before next temperature
            await cleanup_provider(llm)
            await asyncio.sleep(2)
            
        # Check response variation increases with temperature
        for temp in temperatures:
            assert len(set(responses[temp])) > 0, f"No responses for temperature {temp}"
            
        # Higher temperature should give more varied responses
        unique_responses_0 = len(set(responses[0.0]))
        unique_responses_1 = len(set(responses[1.0]))
        assert unique_responses_1 >= unique_responses_0, "Higher temperature not producing more variation"
            
    finally:
        if llm:
            await cleanup_provider(llm)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_limits(api_keys):
    """Test handling of token/length limits"""
    provider = 'openai'  # OpenAI has clear token limits
    llm = None
    try:
        llm = LLMEasy(
            provider=provider,
            api_key=api_keys['OPENAI_API_KEY'],
            max_tokens=10  # Very low limit to test truncation
        )
        
        # Test with a prompt that would generate a long response
        response = await llm.query(
            prompt="Write a detailed essay about Python programming language",
            system="Be comprehensive"
        )
        
        # Response should be truncated
        words = response.split()
        assert len(words) < 20, "Response not properly truncated"
        
    finally:
        if llm:
            await cleanup_provider(llm)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_response_consistency(api_keys):
    """Test response consistency with temperature=0"""
    provider = 'openai'
    llm = None
    try:
        llm = LLMEasy(
            provider=provider,
            api_key=api_keys['OPENAI_API_KEY'],
            temperature=0  # Should give consistent responses
        )
        
        # Get multiple responses to the same prompt
        prompt = "What is 2+2?"
        responses = []
        for _ in range(3):
            response = await llm.query(prompt)
            responses.append(response)
            await asyncio.sleep(1)
            
        # All responses should be identical or very similar
        first_response = responses[0]
        for resp in responses[1:]:
            assert resp.lower().strip() == first_response.lower().strip(), \
                "Inconsistent responses with temperature=0"
            
    finally:
        if llm:
            await cleanup_provider(llm)