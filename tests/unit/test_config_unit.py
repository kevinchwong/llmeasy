import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeasy.utils.config import LLMSettings
import os
import tempfile
import yaml
import logging
from dataclasses import asdict, field, dataclass, fields
from .helpers.settings_impl import MockSettings

logger = logging.getLogger(__name__)

class TestConfiguration:
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file with valid YAML"""
        config_data = {
            'settings': {
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'timeout': 30,
                'stream_chunk_size': 1000,
                'json_repair': True,
                'max_buffer_size': 10000
            },
            'provider_settings': {
                'claude': {
                    'model': 'claude-3-sonnet-20240229',
                    'max_tokens_to_sample': 2000
                },
                'openai': {
                    'model': 'gpt-4-turbo-preview',
                    'response_format': None
                },
                'mistral': {
                    'model': 'mistral-large-latest',
                    'safe_mode': True
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
            temp_path = f.name
            
        # Return path and cleanup after test
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
    def test_settings_loading(self):
        """Test loading settings from dataclass"""
        settings = MockSettings()
        
        # Check default values
        assert 0 <= settings.temperature <= 1
        assert settings.max_tokens > 0
        assert settings.timeout > 0
        assert settings.stream_chunk_size > 0
        assert isinstance(settings.json_repair, bool)
        
    def test_provider_settings(self):
        """Test provider-specific settings"""
        settings = LLMSettings()
        settings_dict = asdict(settings)
        
        # Check Claude settings
        assert settings_dict['claude_model'].startswith('claude-')
        assert settings_dict['claude_max_tokens_to_sample'] > 0
        
        # Check OpenAI settings
        assert settings_dict['openai_model'].startswith('gpt-')
        
        # Check Mistral settings
        assert settings_dict['mistral_model'].startswith('mistral-')
        assert isinstance(settings_dict['mistral_safe_mode'], bool)
        
    def test_invalid_settings(self):
        """Test handling of invalid settings"""
        # Test invalid temperature
        with pytest.raises(ValueError):
            settings = MockSettings()
            settings.temperature = 2.0  # Must be between 0 and 1
            
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            settings = MockSettings()
            settings.max_tokens = -1  # Must be positive
            
        # Test invalid timeout
        with pytest.raises(ValueError):
            settings = MockSettings()
            settings.timeout = 0  # Must be positive
            
    def test_settings_update(self):
        """Test updating settings"""
        settings = MockSettings()
        
        # Update valid settings
        settings.temperature = 0.8
        settings.max_tokens = 1500
        settings.timeout = 45
        
        assert settings.temperature == 0.8
        assert settings.max_tokens == 1500
        assert settings.timeout == 45
        
        # Test invalid updates
        with pytest.raises(ValueError):
            settings.temperature = -0.1  # Must be between 0 and 1
            
        with pytest.raises(ValueError):
            settings.max_tokens = 0  # Must be positive

    def test_provider_validation(self):
        """Test validation of provider-specific settings"""
        settings = LLMSettings()
        
        # Test Claude model validation
        invalid_model = "invalid-model"
        valid_model = "claude-3-sonnet-20240229"
        
        settings.claude_model = valid_model  # Should work
        assert settings.claude_model == valid_model
        
        # Test OpenAI model validation
        valid_openai = "gpt-4-turbo-preview"
        settings.openai_model = valid_openai  # Should work
        assert settings.openai_model == valid_openai

    def test_environment_variables(self, monkeypatch):
        """Test loading settings from environment variables"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        
        from llmeasy.utils.config import get_api_key
        
        assert get_api_key("openai") == "test-key"
        assert get_api_key("claude") == "test-key"
        assert get_api_key("mistral") == "test-key"