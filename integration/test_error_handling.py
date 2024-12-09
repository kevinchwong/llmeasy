import pytest
from llmeasy import LLMEasy
import asyncio
from integration.test_utils import log_test_failure, cleanup_provider
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling(api_keys):
    """Test error handling and rate limiting"""
    # Test invalid provider
    with pytest.raises(ValueError) as exc_info:
        LLMEasy(provider='invalid_provider', api_key='test')
    assert "Unsupported provider" in str(exc_info.value)
    
    # Test rate limiting and retries
    provider_name = 'openai'
    llm = None
    try:
        llm = LLMEasy(
            provider=provider_name,
            api_key=api_keys['OPENAI_API_KEY'],
            temperature=0.7
        )
        
        # Send multiple requests quickly to trigger rate limiting
        responses = []
        for i in range(5):
            try:
                response = await llm.query(
                    prompt=f"Quick test {i}",
                    system="Be very brief."
                )
                responses.append(response)
                await asyncio.sleep(0.1)  # Very short delay to trigger rate limits
            except Exception as e:
                logger.info(f"Expected rate limit error: {e}")
                assert "rate" in str(e).lower() or "limit" in str(e).lower()
                break
                
    finally:
        if llm:
            await cleanup_provider(llm) 

@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests(api_keys):
    """Test handling of concurrent requests"""
    provider = 'claude'  # Claude handles concurrency well
    llm = None
    try:
        llm = LLMEasy(
            provider=provider,
            api_key=api_keys['ANTHROPIC_API_KEY']
        )
        
        # Send multiple requests concurrently
        async def make_request(i: int) -> str:
            return await llm.query(
                prompt=f"Count to {i}",
                system="Be very brief."
            )
            
        tasks = [make_request(i) for i in range(1, 4)]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 3, "Not all concurrent requests completed"
        assert all(isinstance(r, str) for r in responses), "Invalid response type in concurrent requests"
        
    finally:
        if llm:
            await cleanup_provider(llm)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_recovery(api_keys):
    """Test recovery from various error conditions"""
    provider = 'openai'
    llm = None
    try:
        llm = LLMEasy(
            provider=provider,
            api_key=api_keys['OPENAI_API_KEY']
        )
        
        # Test recovery from network timeout
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):  # Very short timeout to force error
                await llm.query("Test prompt")
                
        # Should still work after timeout
        response = await llm.query("Test prompt")
        assert isinstance(response, str)
        
        # Test recovery from invalid API key
        invalid_llm = LLMEasy(
            provider=provider,
            api_key="invalid_key_that_will_fail"
        )
        
        with pytest.raises((ValueError, Exception)):
            await invalid_llm.query("Test prompt")
            
        # Original LLM should still work
        response = await llm.query("Test prompt")
        assert isinstance(response, str)
        
        # Test recovery from invalid model name
        invalid_model_llm = LLMEasy(
            provider=provider,
            api_key=api_keys['OPENAI_API_KEY'],
            model="non_existent_model"
        )
        
        with pytest.raises((ValueError, Exception)):
            await invalid_model_llm.query("Test prompt")
            
        # Original LLM should still work
        response = await llm.query("Test prompt")
        assert isinstance(response, str)
        
    except Exception as e:
        log_test_failure(
            provider,
            e,
            context="During error recovery test"
        )
        raise
        
    finally:
        if llm:
            await cleanup_provider(llm)
        if 'invalid_llm' in locals():
            await cleanup_provider(invalid_llm)
        if 'invalid_model_llm' in locals():
            await cleanup_provider(invalid_model_llm)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_message_localization(api_keys):
    """Test error messages in different languages"""
    provider = 'openai'
    llm = None
    try:
        # Test with invalid API key to trigger error
        llm = LLMEasy(
            provider=provider,
            api_key="invalid_key"
        )
        
        with pytest.raises(Exception) as exc_info:
            await llm.query("Test")
            
        error_msg = str(exc_info.value)
        assert "api" in error_msg.lower() or "key" in error_msg.lower(), "Error message not clear"
        assert error_msg.isprintable(), "Error contains invalid characters"
        
    finally:
        if llm:
            await cleanup_provider(llm)