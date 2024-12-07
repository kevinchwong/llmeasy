import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm_query import LLMQuery
from llm_query.providers import OpenAIProvider, ClaudeProvider

@pytest.mark.asyncio
async def test_query_with_template(mock_openai_client):
    """Test basic template query"""
    with patch('llm_query.providers.openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value = mock_openai_client
        llm = LLMQuery(provider='openai', api_key='test-key')
        template = "Test ${variable}"
        variables = {"variable": "value"}
        
        response = await llm.query(template=template, variables=variables)
        assert response == '{"test": "response"}'

@pytest.mark.asyncio
async def test_stream_response(mock_openai_client, mock_openai_stream_response):
    """Test streaming response"""
    with patch('llm_query.providers.openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.return_value = mock_openai_stream_response
        
        llm = LLMQuery(provider='openai', api_key='test-key')
        template = "Test ${variable}"
        variables = {"variable": "value"}
        
        chunks = []
        generator = await llm.stream(template=template, variables=variables)
        async for chunk in generator:
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_json_output(mock_openai_client):
    """Test JSON output format"""
    with patch('llm_query.providers.openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value = mock_openai_client
        llm = LLMQuery(provider='openai', api_key='test-key')
        template = "Test ${variable}"
        variables = {"variable": "value"}
        
        response = await llm.query(
            template=template,
            variables=variables,
            output_format='json'
        )
        assert isinstance(response, dict)
        assert response == {"test": "response"}

@pytest.mark.asyncio
async def test_provider_selection():
    """Test provider selection"""
    with patch('llm_query.utils.config.settings', create=True) as mock_settings:
        mock_settings.anthropic_api_key = 'test-key'
        mock_settings.openai_api_key = 'test-key'
        
        claude = LLMQuery(provider='claude', api_key='test-key')
        assert isinstance(claude.provider, ClaudeProvider)
        
        openai = LLMQuery(provider='openai', api_key='test-key')
        assert isinstance(openai.provider, OpenAIProvider)

@pytest.mark.asyncio
async def test_invalid_provider():
    """Test invalid provider handling"""
    with pytest.raises(ValueError) as exc_info:
        LLMQuery(provider='invalid', api_key='test-key')
    assert "Unsupported provider" in str(exc_info.value)

@pytest.mark.asyncio
async def test_template_variables():
    """Test template variable substitution"""
    with patch('llm_query.providers.openai.AsyncOpenAI') as mock_openai:
        # Create mock client and response
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Response with variables"
                    )
                )
            ]
        ))
        mock_openai.return_value = mock_client
        
        llm = LLMQuery(provider='openai', api_key='test-key')
        template = "Hello ${name}, you are ${age} years old"
        variables = {"name": "Alice", "age": "30"}
        
        await llm.query(template=template, variables=variables)
        
        # Verify the formatted prompt
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) == 1
        
        # Check that variables were substituted correctly
        call_kwargs = calls[0][1]  # Get kwargs from first call
        messages = call_kwargs['messages']
        assert len(messages) == 1
        assert "Hello Alice, you are 30 years old" in messages[0]['content']

@pytest.mark.asyncio
async def test_missing_template_variable():
    """Test handling of missing template variables"""
    llm = LLMQuery(provider='openai', api_key='test-key')
    template = "Hello ${name}, you are ${age} years old"
    variables = {"name": "Alice"}  # Missing 'age' variable
    
    with pytest.raises(ValueError) as exc_info:
        await llm.query(template=template, variables=variables)
    assert "Missing required template variable: 'age'" in str(exc_info.value)
