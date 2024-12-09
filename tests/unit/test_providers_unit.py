import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.unit.test_base import BaseLLMTest
from llmeasy.providers import OpenAIProvider, ClaudeProvider, MistralProvider
import logging
import json
from typing import AsyncIterator

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
        
        mock_openai_client.chat.completions.create.return_value = AsyncMockIterator(mock_chunks)
        
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
