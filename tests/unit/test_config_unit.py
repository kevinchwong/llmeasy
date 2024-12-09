import pytest
from tests.unit.test_base import BaseLLMTest
from llmeasy.utils.config import ConfigManager
import os
import tempfile
import yaml

class TestConfiguration(BaseLLMTest):
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file with valid YAML"""
        config_data = {
            'api_keys': {
                'openai': 'test_key',
                'claude': 'test_key'
            },
            'settings': {
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
            temp_path = f.name
            
        # Return path and cleanup after test
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
    def test_config_loading(self, temp_config_file):
        """Test loading configuration from file"""
        config = ConfigManager(config_path=temp_config_file)
        assert config.get_api_key('openai') == 'test_key'
        assert config.settings.temperature == 0.7
        
    def test_config_saving(self, temp_config_file):
        """Test saving configuration changes"""
        config = ConfigManager(config_path=temp_config_file)
        config.set_api_key('mistral', 'new_key')
        config.update_settings(temperature=0.8)
        
        # Load config again to verify changes
        new_config = ConfigManager(config_path=temp_config_file)
        assert new_config.get_api_key('mistral') == 'new_key'
        assert new_config.settings.temperature == 0.8
        
    def test_invalid_config(self):
        """Test handling of invalid config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write valid but empty YAML
            f.write("---\n")
            f.flush()
            temp_path = f.name
            
        try:
            config = ConfigManager(config_path=temp_path)
            assert config.config == {}
            assert config.settings is not None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_missing_config(self):
        """Test handling of missing config file"""
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        assert config.config == {}
        assert config.settings is not None
        
    def test_config_defaults(self):
        """Test default configuration values"""
        config = ConfigManager()
        assert config.settings.temperature >= 0 and config.settings.temperature <= 1
        assert config.settings.max_tokens > 0