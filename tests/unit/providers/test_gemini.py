import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeasy.providers import GeminiProvider

@pytest.mark.asyncio
async def test_gemini_query(mock_gemini_client):
    """Test basic Gemini query"""
    provider = GeminiProvider('test-key')
    provider.client = mock_gemini_client
    
    mock_gemini_client.generate_content_async.return_value = MagicMock(
        text="Test response"
    )
    
    response = await provider._generate_response("Test prompt")
    assert response == "Test response"

@pytest.mark.asyncio
async def test_gemini_json_output(mock_gemini_client):
    """Test Gemini JSON output"""
    provider = GeminiProvider('test-key')
    provider.client = mock_gemini_client
    
    mock_gemini_client.generate_content_async.return_value = MagicMock(
        text='{"colors": ["red", "blue", "green"]}'
    )
    
    response = await provider.query(
        prompt="Test prompt",
        output_format='json'
    )
    assert isinstance(response, dict)
    assert "colors" in response
    assert len(response["colors"]) == 3

@pytest.mark.asyncio
async def test_gemini_stream(mock_gemini_client, mock_gemini_stream_response):
    """Test Gemini streaming"""
    provider = GeminiProvider('test-key')
    
    # Set up streaming mock
    mock_gemini_client.generate_content_async.return_value = mock_gemini_stream_response
    provider.client = mock_gemini_client
    
    # Test streaming response
    stream = await provider._generate_response("Test prompt", stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 2
    assert chunks == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_gemini_error_handling(mock_gemini_client):
    """Test Gemini error handling"""
    provider = GeminiProvider('test-key')
    mock_gemini_client.generate_content_async.side_effect = Exception("Test error")
    provider.client = mock_gemini_client
    
    with pytest.raises(ValueError) as exc_info:
        await provider.query("Test prompt")
    assert "Error generating Gemini response" in str(exc_info.value) 