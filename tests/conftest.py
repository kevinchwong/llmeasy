import pytest
from unittest.mock import AsyncMock, MagicMock
from llm_query import LLMQuery
from typing import AsyncIterator, List, Any

class AsyncIteratorMock:
    """Mock async iterator for streaming responses"""
    def __init__(self, items: List[Any]):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item

@pytest.fixture
def mock_openai_stream_response():
    """Mock streaming response from OpenAI"""
    return AsyncIteratorMock([
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))])
    ])

@pytest.fixture
def mock_claude_stream_response():
    """Mock streaming response from Claude"""
    return AsyncIteratorMock([
        MagicMock(delta=MagicMock(text="chunk1")),
        MagicMock(delta=MagicMock(text="chunk2"))
    ])

@pytest.fixture
def mock_openai_response():
    """Mock standard OpenAI response"""
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"test": "response"}'
                )
            )
        ]
    )

@pytest.fixture
def mock_claude_response():
    """Mock standard Claude response"""
    return MagicMock(
        content=[
            MagicMock(
                text='{"test": "response"}'
            )
        ]
    )

@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client"""
    client = MagicMock()
    create_mock = AsyncMock()
    create_mock.return_value = mock_openai_response
    create_mock.side_effect = None
    client.chat.completions.create = create_mock
    return client

@pytest.fixture
def mock_claude_client(mock_claude_response):
    """Mock Claude client"""
    client = MagicMock()
    create_mock = AsyncMock()
    create_mock.return_value = mock_claude_response
    create_mock.side_effect = None
    client.messages.create = create_mock
    return client

@pytest.fixture
def mock_settings():
    """Mock settings with no API keys"""
    return MagicMock(
        anthropic_api_key=None,
        openai_api_key=None,
        claude_model="claude-3-sonnet-20240229",
        openai_model="gpt-4-turbo-preview",
        max_tokens=1000,
        temperature=0.7
    ) 