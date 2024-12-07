import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeasy.providers import MistralProvider

@pytest.mark.asyncio
async def test_mistral_query(mock_mistral_client):
    """Test basic Mistral query"""
    provider = MistralProvider('test-key')
    provider.client = mock_mistral_client
    
    mock_mistral_client.chat_async.return_value = MagicMock(
        content="Test response"
    )
    
    response = await provider._generate_response("Test prompt")
    assert response == "Test response"

@pytest.mark.asyncio
async def test_mistral_json_output(mock_mistral_client):
    """Test Mistral JSON output"""
    provider = MistralProvider('test-key')
    provider.client = mock_mistral_client
    
    mock_mistral_client.chat_async.return_value = MagicMock(
        content='{"colors": ["red", "blue", "green"]}'
    )
    
    response = await provider.query(
        prompt="Test prompt",
        output_format='json'
    )
    assert isinstance(response, dict)
    assert "colors" in response
    assert len(response["colors"]) == 3

@pytest.mark.asyncio
async def test_mistral_stream(mock_mistral_client, mock_mistral_stream_response):
    """Test Mistral streaming"""
    provider = MistralProvider('test-key')
    
    # Set up streaming mock
    mock_mistral_client.chat_async.return_value = mock_mistral_stream_response
    provider.client = mock_mistral_client
    
    # Test streaming response
    stream = await provider._generate_response("Test prompt", stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 2
    assert chunks == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_mistral_error_handling(mock_mistral_client):
    """Test Mistral error handling"""
    provider = MistralProvider('test-key')
    mock_mistral_client.chat_async.side_effect = Exception("Test error")
    provider.client = mock_mistral_client
    
    with pytest.raises(ValueError) as exc_info:
        await provider.query("Test prompt")
    assert "Error generating Mistral response" in str(exc_info.value) 