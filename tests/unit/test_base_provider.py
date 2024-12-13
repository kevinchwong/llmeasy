import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import logging
from typing import Optional, Any
from .helpers.provider_impl import MockProvider

logger = logging.getLogger(__name__)

class TestBaseProvider:
    def test_initialization(self):
        """Test provider initialization"""
        provider = MockProvider(api_key="test_key", temperature=0.5)
        assert provider.api_key == "test_key"
        assert provider.temperature == 0.5
        
    def test_parse_response(self):
        """Test response parsing"""
        provider = MockProvider(api_key="test_key")
        
        # Test JSON parsing
        json_str = '{"test": "value"}'
        parsed = provider._parse_response(json_str, 'json')
        assert isinstance(parsed, dict)
        assert parsed["test"] == "value"
        
        # Test non-JSON response
        response = provider._parse_response("test", None)
        assert response == "test"
        
    def test_validate_response(self):
        """Test response validation"""
        provider = MockProvider(api_key="test_key")
        
        # Test JSON validation
        assert provider.validate_response('{"test": "value"}', 'json')
        assert not provider.validate_response('invalid json', 'json')
        
        # Test non-JSON validation
        assert provider.validate_response("test", None)
        
    @pytest.mark.asyncio
    async def test_query_with_format(self):
        """Test query with formatting"""
        provider = MockProvider(api_key="test_key")
        
        # Test JSON output
        response = await provider.query("Test", output_format='json')
        assert isinstance(response, str)
        assert response == "Test response"
            
    def test_format_prompt(self):
        """Test prompt formatting"""
        provider = MockProvider(api_key="test_key")
        
        # Test basic formatting
        formatted = provider._format_prompt("Test prompt", None)
        assert formatted == "Test prompt"
        
        # Test JSON format
        formatted = provider._format_prompt("Test prompt", 'json')
        assert "json" in formatted.lower()
        