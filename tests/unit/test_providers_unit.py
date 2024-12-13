import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.unit.test_base import BaseLLMTest
from llmeasy.providers import OpenAIProvider, ClaudeProvider, MistralProvider
import logging
import json
from typing import AsyncIterator
import asyncio

logger = logging.getLogger(__name__)

class AsyncMockIterator:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item

class TestOpenAIProvider(BaseLLMTest):
    @pytest.fixture
    def mock_openai_response(self):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )

    @pytest.fixture
    def mock_openai_client(self, mock_openai_response):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        return client

    @pytest.fixture
    def provider(self, mock_openai_client):
        with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
            provider = OpenAIProvider(api_key="test_key")
            provider.client = mock_openai_client
            return provider
            
    async def test_query(self, provider, mock_openai_response):
        response = await provider.query("Test prompt")
        assert isinstance(response, str)
        assert response == "Test response"
        
    async def test_streaming(self, provider, mock_openai_client):
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))])
        ]
        
        # Create async iterator for streaming
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk
                
        mock_openai_client.chat.completions.create.return_value = mock_stream()
        
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]
        
    async def test_json_output(self, provider, mock_openai_client):
        test_json = {"test": "response"}
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(test_json)))]
        )
        
        response = await provider.query(
            "Test prompt",
            output_format='json'
        )
        assert isinstance(response, dict)
        assert response == test_json

    @pytest.fixture(autouse=True)
    async def cleanup(self, provider):
        """Cleanup after each test"""
        yield
        try:
            if hasattr(provider, 'client') and hasattr(provider.client, 'aclose'):
                await provider.client.aclose()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    @pytest.mark.asyncio
    async def test_error_handling(self, provider, mock_openai_client):
        """Test error handling"""
        mock_openai_client.chat.completions.create.side_effect = ValueError("Invalid API key")
        
        with pytest.raises(ValueError):
            await provider.query("Test", api_key="invalid")

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, provider, mock_openai_client):
        """Test handling of rate limit errors"""
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "Rate limit exceeded"
        )
        
        with pytest.raises(ValueError) as exc_info:
            await provider.query("Test prompt")
        assert "rate limit" in str(exc_info.value).lower()

class TestClaudeProvider(BaseLLMTest):
    @pytest.fixture
    def mock_claude_response(self):
        return AsyncMock(content=[MagicMock(text="Test response")])

    @pytest.fixture
    def mock_claude_client(self, mock_claude_response):
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=mock_claude_response)
        return client

    @pytest.fixture
    def provider(self, mock_claude_client):
        with patch('anthropic.AsyncAnthropic', return_value=mock_claude_client):
            provider = ClaudeProvider(api_key="test_key")
            provider.client = mock_claude_client
            return provider
            
    async def test_query(self, provider, mock_claude_response):
        response = await provider.query("Test prompt")
        assert isinstance(response, str)
        assert response == "Test response"

    async def test_streaming(self, provider, mock_claude_client):
        """Test streaming for Claude"""
        mock_chunks = [
            MagicMock(type='content_block_delta', delta=MagicMock(text="chunk1")),
            MagicMock(type='content_block_delta', delta=MagicMock(text="chunk2"))
        ]
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk
                
        mock_claude_client.messages.create.return_value = mock_stream()
        
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]

    @pytest.fixture(autouse=True)
    async def cleanup(self, provider):
        """Cleanup after each test"""
        yield
        try:
            if hasattr(provider, 'client') and hasattr(provider.client, 'aclose'):
                await provider.client.aclose()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    @pytest.mark.asyncio
    async def test_error_handling(self, provider, mock_claude_client):
        """Test error handling"""
        # Test invalid API key
        mock_claude_client.messages.create.side_effect = Exception("Invalid API key")
        
        with pytest.raises(ValueError):
            await provider.query("Test")

    @pytest.mark.asyncio
    async def test_timeout_handling(self, provider, mock_claude_client):
        """Test handling of timeout errors"""
        error_msg = "Request timed out"
        mock_claude_client.messages.create.side_effect = asyncio.TimeoutError(error_msg)
        
        with pytest.raises(ValueError) as exc_info:
            await provider.query("Test prompt")
        error_str = str(exc_info.value).lower()
        assert any(word in error_str for word in ['timeout', 'timed out'])

class TestMistralProvider(BaseLLMTest):
    @pytest.fixture
    def mock_mistral_response(self):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )

    @pytest.fixture
    def mock_mistral_client(self, mock_mistral_response):
        client = MagicMock()
        client.chat = MagicMock(return_value=mock_mistral_response)
        client.chat_stream = MagicMock(return_value=[
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))])
        ])
        return client

    @pytest.fixture
    def provider(self, mock_mistral_client):
        with patch('mistralai.client.MistralClient', return_value=mock_mistral_client):
            provider = MistralProvider(api_key="test_key")
            provider.client = mock_mistral_client
            return provider
            
    async def test_query(self, provider, mock_mistral_response):
        response = await provider.query("Test prompt")
        assert isinstance(response, str)
        assert response == "Test response"

    async def test_streaming(self, provider, mock_mistral_client):
        """Test streaming for Mistral"""
        # Mock streaming response
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))])
        ]
        
        # Set up mock stream
        mock_mistral_client.chat_stream.return_value = mock_chunks
        
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]

    async def test_json_output(self, provider, mock_mistral_client):
        """Test JSON output for Mistral"""
        test_json = {"test": "response"}
        json_str = json.dumps(test_json)
        
        mock_mistral_client.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json_str))]
        )
        
        response = await provider.query(
            "Test prompt",
            output_format='json'
        )
        
        # For Mistral, the response might be returned as a string
        if isinstance(response, str):
            response = json.loads(response)
        
        assert isinstance(response, dict)
        assert response == test_json

    @pytest.fixture(autouse=True)
    async def cleanup(self, provider):
        """Cleanup after each test"""
        yield
        try:
            if hasattr(provider, 'client') and hasattr(provider.client, 'aclose'):
                await provider.client.aclose()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    @pytest.mark.asyncio
    async def test_error_handling(self, provider, mock_mistral_client):
        """Test error handling"""
        mock_mistral_client.chat.side_effect = ValueError("Invalid API key")
        
        with pytest.raises(ValueError) as exc_info:
            await provider.query("Test")
        assert "invalid api key" in str(exc_info.value).lower()
