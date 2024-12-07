import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeasy.providers import ClaudeProvider
from tests.conftest import AsyncIteratorMock

@pytest.mark.asyncio
async def test_claude_query(mock_claude_client):
    """Test basic Claude query"""
    provider = ClaudeProvider('test-key')
    provider.client = mock_claude_client
    
    response = await provider._generate_response("Test prompt")
    assert response == '{"test": "response"}'

@pytest.mark.asyncio
async def test_claude_json_output(mock_claude_client):
    """Test Claude JSON output"""
    provider = ClaudeProvider('test-key')
    provider.client = mock_claude_client
    
    mock_claude_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"colors": ["red", "blue", "green"]}')]
    )
    
    response = await provider.query(
        prompt="Test prompt",
        output_format='json'
    )
    assert isinstance(response, dict)
    assert "colors" in response
    assert len(response["colors"]) == 3

@pytest.mark.asyncio
async def test_claude_stream(mock_claude_client, mock_claude_stream_response):
    """Test Claude streaming"""
    provider = ClaudeProvider('test-key')
    
    # Set up streaming mock
    mock_claude_client.messages.create.return_value = AsyncIteratorMock([
        MagicMock(delta=MagicMock(text="chunk1")),
        MagicMock(delta=MagicMock(text="chunk2"))
    ])
    provider.client = mock_claude_client
    
    # Test streaming response
    stream = await provider._generate_response("Test prompt", stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 2
    assert chunks == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_claude_error_handling(mock_claude_client):
    """Test Claude error handling"""
    provider = ClaudeProvider('test-key')
    mock_claude_client.messages.create.side_effect = Exception("Test error")
    provider.client = mock_claude_client
    
    with pytest.raises(ValueError) as exc_info:
        await provider.query("Test prompt")
    assert "Error generating Claude response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_claude_format_prompt():
    """Test prompt formatting"""
    provider = ClaudeProvider('test-key')
    
    # Test JSON format
    json_prompt = provider._format_prompt("Test", "json")
    assert "JSON format" in json_prompt
    assert "Do not include any explanatory text" in json_prompt
    
    # Test other format
    other_prompt = provider._format_prompt("Test", "markdown")
    assert "markdown format" in other_prompt

@pytest.mark.asyncio
async def test_claude_validate_response():
    """Test response validation"""
    provider = ClaudeProvider('test-key')
    
    # Test valid JSON with markdown formatting
    assert provider.validate_response('```json\n{"test": true}\n```', "json")
    
    # Test valid JSON without markdown
    assert provider.validate_response('{"test": true}', "json")
    
    # Test invalid JSON
    assert not provider.validate_response('invalid json', "json")
    
    # Test non-JSON
    assert provider.validate_response("any text", None)

@pytest.mark.asyncio
async def test_claude_parse_response():
    """Test response parsing"""
    provider = ClaudeProvider('test-key')
    
    # Test JSON with markdown
    json_response = provider._parse_response('```json\n{"test": true}\n```', "json")
    assert isinstance(json_response, dict)
    assert json_response["test"] is True
    
    # Test JSON without markdown
    json_response = provider._parse_response('{"test": true}', "json")
    assert isinstance(json_response, dict)
    assert json_response["test"] is True
    
    # Test non-JSON
    text_response = provider._parse_response("Hello world", None)
    assert text_response == "Hello world"
    
    # Test invalid JSON
    with pytest.raises(ValueError) as exc_info:
        provider._parse_response("invalid json", "json")
    assert "Invalid JSON response" in str(exc_info.value)