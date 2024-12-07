import warnings
import pytest
from unittest.mock import AsyncMock, MagicMock
from llmeasy import LLMEasy
from typing import AsyncIterator, List, Any

# Filter out all Google Protobuf related deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="google._upb._message"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*PyType_Spec.*"
)

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

@pytest.fixture
def mock_gemini_stream_response():
    """Mock streaming response from Gemini"""
    return AsyncIteratorMock([
        MagicMock(text="chunk1"),
        MagicMock(text="chunk2")
    ])

@pytest.fixture
def mock_mistral_stream_response():
    """Mock streaming response from Mistral"""
    return AsyncIteratorMock([
        MagicMock(delta="chunk1"),
        MagicMock(delta="chunk2")
    ])

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client"""
    client = MagicMock()
    generate_mock = AsyncMock()
    generate_mock.return_value = MagicMock(text='{"test": "response"}')
    generate_mock.side_effect = None
    client.generate_content_async = generate_mock
    return client

@pytest.fixture
def mock_mistral_client():
    """Mock Mistral client"""
    client = MagicMock()
    chat_mock = AsyncMock()
    chat_mock.return_value = MagicMock(content='{"test": "response"}')
    chat_mock.side_effect = None
    client.chat_async = chat_mock
    return client
