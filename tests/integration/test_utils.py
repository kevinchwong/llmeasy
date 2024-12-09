import logging
import traceback
from datetime import datetime
from asyncio import TimeoutError
from llmeasy import LLMEasy
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

def retry_on_exception(retries=3, delay=1):
    """Decorator to retry flaky tests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i < retries - 1:
                        logger.warning(f"Attempt {i+1} failed, retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def log_test_failure(provider: str, error: Exception, context: str = None):
    """Log detailed test failure information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_msg = [
        f"\n{'='*50}",
        f"Test Failure Details - {timestamp}",
        f"{'='*50}",
        f"Provider: {provider}",
        f"Error Type: {type(error).__name__}",
        f"Error Message: {str(error)}",
    ]
    
    if context:
        error_msg.append(f"Context: {context}")
    
    error_msg.extend([
        "Traceback:",
        traceback.format_exc(),
        f"{'='*50}\n"
    ])
    
    logger.error('\n'.join(error_msg))

async def cleanup_provider(llm: LLMEasy):
    """Cleanup provider resources with timeout"""
    try:
        async with asyncio.timeout(5):
            if hasattr(llm.provider, 'client') and hasattr(llm.provider.client, 'aclose'):
                await llm.provider.client.aclose()
    except TimeoutError:
        logger.warning("Provider cleanup timed out")
    except Exception as e:
        logger.warning(f"Error during provider cleanup: {e}")

def validate_response_content(response: str, provider: str) -> None:
    """Validate response content quality"""
    min_words = 5
    max_words = 1000
    words = response.split()
    
    assert len(words) >= min_words, f"Response from {provider} too short: {len(words)} words"
    assert len(words) <= max_words, f"Response from {provider} too long: {len(words)} words"
    assert not any(len(word) > 50 for word in words), f"Found suspiciously long word in {provider} response"

async def check_provider_health(llm: LLMEasy) -> bool:
    """Utility function to check provider health"""
    try:
        response = await llm.query(
            prompt="Respond with OK",
            system="You are a test system. Respond with exactly 'OK'."
        )
        return bool(response and 'ok' in response.lower().split())
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False

def validate_model_response(response: str, min_words: int = 5, max_words: int = 1000) -> None:
    """Validate model response meets basic quality criteria"""
    words = response.split()
    assert len(words) >= min_words, f"Response too short: {len(words)} words"
    assert len(words) <= max_words, f"Response too long: {len(words)} words"
    assert not any(len(word) > 50 for word in words), "Found suspiciously long word"
    assert not any(word.count('.') > 5 for word in words), "Found suspicious repetition"

def validate_streaming_chunks(chunks: list) -> None:
    """Validate streaming response chunks"""
    assert len(chunks) > 0, "No chunks received"
    assert all(isinstance(c, str) for c in chunks), "Invalid chunk type"
    assert all(len(c) > 0 for c in chunks), "Empty chunk found"
    complete = ''.join(chunks)
    assert len(complete) > 0, "Empty complete response"
    validate_model_response(complete)

def validate_response_quality(response: str, provider: str) -> None:
    """Validate response quality and content"""
    # Basic validation
    assert isinstance(response, str), f"Response from {provider} is not a string"
    assert len(response) > 0, f"Empty response from {provider}"
    
    # Content validation
    words = response.split()
    assert len(words) >= 5, f"Response from {provider} too short: {len(words)} words"
    assert len(words) <= 1000, f"Response from {provider} too long: {len(words)} words"
    
    # Quality checks
    assert not any(len(word) > 50 for word in words), f"Found suspiciously long word in {provider} response"
    assert not any(word.count('.') > 5 for word in words), f"Found suspicious repetition in {provider} response"
    assert not response.count('?') > 10, f"Too many questions in {provider} response"