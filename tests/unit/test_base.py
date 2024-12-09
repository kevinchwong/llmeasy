import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeasy import LLMEasy
import json
import logging

logger = logging.getLogger(__name__)

async def cleanup_provider(self, llm: LLMEasy):
    """Cleanup provider resources with timeout"""
    try:
        async with asyncio.timeout(5):
            if hasattr(llm.provider, 'client') and hasattr(llm.provider.client, 'aclose'):
                await llm.provider.client.aclose()
    except TimeoutError:
        logger.warning("Provider cleanup timed out")
    except Exception as e:
        logger.warning(f"Error during provider cleanup: {e}")


class BaseLLMTest:
    """Base class for LLM tests with common mocks"""
    
    @pytest.fixture
    def mock_response(self):
        return {
            "test": "response",
            "number": 42,
            "array": ["item1", "item2"]
        }
        
    @pytest.fixture
    def mock_stream_chunks(self):
        return ["chunk1", "chunk2", "chunk3"]
        
    @pytest.fixture
    def mock_error_response(self):
        return MagicMock(
            status_code=429,
            text="Rate limit exceeded"
        )
        
    def assert_json_response(self, response, expected):
        """Helper to assert JSON responses"""
        assert isinstance(response, dict)
        assert response == expected
        
    def assert_stream_chunks(self, chunks, expected):
        """Helper to assert streaming chunks"""
        assert list(chunks) == expected 
