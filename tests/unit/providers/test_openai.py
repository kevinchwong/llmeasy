import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm_query.providers import OpenAIProvider

@pytest.mark.asyncio
async def test_openai_query(mock_openai_client):
    """Test basic OpenAI query"""
    provider = OpenAIProvider('test-key')
    provider.client = mock_openai_client
    
    response = await provider._generate_response("Test prompt")
    assert response == '{"test": "response"}'

@pytest.mark.asyncio
async def test_openai_json_output(mock_openai_client):
    """Test OpenAI JSON output"""
    provider = OpenAIProvider('test-key')
    provider.client = mock_openai_client
    
    response = await provider.query(
        prompt="Test prompt",
        output_format='json'
    )
    assert isinstance(response, dict)
    assert response == {"test": "response"}

@pytest.mark.asyncio
async def test_openai_stream(mock_openai_client, mock_openai_stream_response):
    """Test OpenAI streaming"""
    provider = OpenAIProvider('test-key')
    
    # Set up streaming mock
    mock_openai_client.chat.completions.create.return_value = mock_openai_stream_response
    provider.client = mock_openai_client
    
    # Test streaming response
    stream = await provider._generate_response("Test prompt", stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 2
    assert chunks == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_openai_error_handling(mock_openai_client):
    """Test OpenAI error handling"""
    provider = OpenAIProvider('test-key')
    mock_openai_client.chat.completions.create.side_effect = Exception("Test error")
    provider.client = mock_openai_client
    
    with pytest.raises(ValueError) as exc_info:
        await provider.query("Test prompt")
    assert "Error generating OpenAI response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_openai_format_prompt():
    """Test prompt formatting"""
    provider = OpenAIProvider('test-key')
    
    # Test JSON format
    json_prompt = provider._format_prompt("Test", "json")
    assert "JSON" in json_prompt
    
    # Test other format
    other_prompt = provider._format_prompt("Test", "markdown")
    assert "markdown format" in other_prompt

@pytest.mark.asyncio
async def test_openai_validate_response():
    """Test response validation"""
    provider = OpenAIProvider('test-key')
    
    # Test valid JSON
    assert provider.validate_response('{"test": true}', "json")
    
    # Test invalid JSON
    assert not provider.validate_response('invalid json', "json")
    
    # Test non-JSON
    assert provider.validate_response("any text", None) 