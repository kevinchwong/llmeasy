import pytest
from llmeasy import LLMEasy
import asyncio
from integration.test_utils import log_test_failure, cleanup_provider, retry_on_exception, validate_response_quality
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
@retry_on_exception(retries=3, delay=2)
async def test_streaming_all_providers(http_client, api_keys):
    """Test streaming functionality for all providers"""
    providers = {
        'claude': {
            'api_key': api_keys['ANTHROPIC_API_KEY'],
            'expected_reliability': True,
            'timeout': 30,
            'retry_attempts': 2
        },
        'openai': {
            'api_key': api_keys['OPENAI_API_KEY'],
            'expected_reliability': True,
            'timeout': 30,
            'retry_attempts': 2
        },
        'mistral': {
            'api_key': api_keys['MISTRAL_API_KEY'],
            'expected_reliability': False,
            'timeout': 40,
            'retry_attempts': 3
        }
    }
    
    for provider_name, config in providers.items():
        if not config['api_key']:
            continue
            
        logger.info(f"\nTesting streaming for {provider_name}")
        llm = None
        chunks_received = 0
        valid_chunks = 0
        retry_count = 0
        
        while retry_count <= config['retry_attempts']:
            try:
                llm = LLMEasy(provider=provider_name, api_key=config['api_key'])
                chunks = []
                
                try:
                    async with asyncio.timeout(config['timeout']):
                        async for chunk in llm.stream(
                            prompt="Explain async/await in Python",
                            system="You are a Python expert."
                        ):
                            chunks_received += 1
                            logger.debug(f"Received chunk {chunks_received} from {provider_name}")
                            
                            try:
                                # Basic validation
                                assert isinstance(chunk, str), f"Invalid chunk type: {type(chunk)}"
                                
                                # Clean and validate chunk
                                cleaned_chunk = ''.join(c for c in chunk if c.isprintable())
                                if cleaned_chunk.strip():  # Only append non-empty chunks
                                    chunks.append(cleaned_chunk)
                                    valid_chunks += 1
                                    logger.debug(f"Valid chunk {valid_chunks}: {cleaned_chunk[:50]}...")
                                else:
                                    logger.debug(f"Skipping empty chunk {chunks_received}")
                                    
                            except AssertionError as e:
                                log_test_failure(
                                    provider_name,
                                    e,
                                    context=f"Chunk {chunks_received}: {chunk!r}"
                                )
                                raise
                            
                    # If we get here, streaming was successful
                    break
                            
                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count <= config['retry_attempts']:
                        delay = 2 ** retry_count  # Exponential backoff
                        logger.warning(f"Streaming timeout for {provider_name}, retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        log_test_failure(
                            provider_name,
                            TimeoutError("Streaming timeout after all retries"),
                            context=f"Chunks received: {chunks_received}, Valid chunks: {valid_chunks}"
                        )
                        raise
                except Exception as e:
                    log_test_failure(
                        provider_name,
                        e,
                        context=f"During streaming operation. Chunks received: {chunks_received}, Valid chunks: {valid_chunks}"
                    )
                    raise
                    
                # Ensure we received some valid chunks
                assert valid_chunks > 0, f"No valid chunks received from {provider_name}"
                
                complete_response = ''.join(chunks)
                assert len(complete_response.strip()) > 0, f"Empty complete response from {provider_name}"
                
                # Validate the complete response
                validate_response_quality(complete_response, provider_name)
                
                logger.info(f"✅ Successfully tested streaming for {provider_name}")
                logger.info(f"Received {chunks_received} chunks ({valid_chunks} valid)")
                logger.debug(f"Complete response preview: {complete_response[:100]}...")
                
            except Exception as e:
                if not config['expected_reliability']:
                    logger.warning(f"Expected failure for {provider_name}: {e}")
                else:
                    pytest.fail(f"Streaming test failed for {provider_name}: {e}")
            finally:
                if llm:
                    await cleanup_provider(llm)
                    await asyncio.sleep(1)