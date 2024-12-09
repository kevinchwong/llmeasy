from typing import Optional, Dict, Any
import yaml
import os
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMSettings:
    """Settings for LLM configuration"""
    # General settings
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    stream_chunk_size: int = 1000
    json_repair: bool = True
    max_buffer_size: int = 10000
    
    # Provider-specific settings
    # Claude settings
    claude_model: str = "claude-3-sonnet-20240229"
    claude_max_tokens_to_sample: int = 2000
    
    # OpenAI settings
    openai_model: str = "gpt-4-turbo-preview"
    openai_response_format: Optional[Dict] = None
    
    # Mistral settings
    mistral_model: str = "mistral-large-latest"
    mistral_safe_mode: bool = True
    
    # Gemini settings
    gemini_model: str = "gemini-pro"
    gemini_safety_settings: Optional[Dict] = None
    
    # Grok settings
    grok_model: str = "grok-1"
    grok_safe_prompt: bool = True

    def get_provider_settings(self, provider: str) -> Dict[str, Any]:
        """Get settings specific to a provider"""
        provider_settings = {
            'claude': {
                'model': self.claude_model,
                'max_tokens_to_sample': self.claude_max_tokens_to_sample,
            },
            'openai': {
                'model': self.openai_model,
                'response_format': self.openai_response_format,
            },
            'mistral': {
                'model': self.mistral_model,
                'safe_mode': self.mistral_safe_mode,
            },
            'gemini': {
                'model': self.gemini_model,
                'safety_settings': self.gemini_safety_settings,
            },
            'grok': {
                'model': self.grok_model,
                'safe_prompt': self.grok_safe_prompt,
            }
        }
        return provider_settings.get(provider, {})

    def get_common_settings(self) -> Dict[str, Any]:
        """Get common settings applicable to all providers"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
        }

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.expanduser("~/.llmeasy/config.yaml")
        self.config = self._load_config()
        self.settings = self._load_settings()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            return {}
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_settings(self) -> LLMSettings:
        """Load LLM settings from config"""
        settings_dict = self.config.get('settings', {})
        return LLMSettings(**{
            k: v for k, v in settings_dict.items()
            if k in LLMSettings.__dataclass_fields__
        })
    
    def save_config(self):
        """Save current configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Update settings in config
        self.config['settings'] = asdict(self.settings)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider"""
        return self.config.get('api_keys', {}).get(provider)
        
    def set_api_key(self, provider: str, key: str):
        """Set API key for provider"""
        if 'api_keys' not in self.config:
            self.config['api_keys'] = {}
        self.config['api_keys'][provider] = key
        self.save_config()
    
    def update_settings(self, **kwargs):
        """Update LLM settings"""
        current_settings = asdict(self.settings)
        current_settings.update({
            k: v for k, v in kwargs.items()
            if k in LLMSettings.__dataclass_fields__
        })
        self.settings = LLMSettings(**current_settings)
        self.save_config()
    
    def get_settings(self) -> LLMSettings:
        """Get current LLM settings"""
        return self.settings

# Create a default settings instance
settings = LLMSettings()

# Initialize default config manager
default_config = ConfigManager()
if default_config.settings:
    settings = default_config.settings

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate API key
    if 'api_key' not in config or not config['api_key']:
        raise ValueError("API key is required")
        
    # Validate provider if specified
    if 'provider' in config:
        provider = config['provider'].lower()
        valid_providers = {'openai', 'claude', 'mistral', 'gemini', 'grok'}
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
            
        # Validate model if specified
        if 'model' in config:
            model = config['model']
            # Provider-specific model validation
            if provider == 'openai':
                valid_models = {'gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo'}
                if model not in valid_models:
                    raise ValueError(f"Invalid OpenAI model. Must be one of: {', '.join(valid_models)}")
            elif provider == 'claude':
                valid_models = {'claude-3-opus-20240229', 'claude-3-sonnet-20240229'}
                if model not in valid_models:
                    raise ValueError(f"Invalid Claude model. Must be one of: {', '.join(valid_models)}")
            elif provider == 'mistral':
                valid_models = {'mistral-tiny', 'mistral-small', 'mistral-medium'}
                if model not in valid_models:
                    raise ValueError(f"Invalid Mistral model. Must be one of: {', '.join(valid_models)}")
    
    # Validate temperature if specified
    if 'temperature' in config:
        temp = config['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
            raise ValueError("Temperature must be a float between 0 and 1")
            
    # Validate max_tokens if specified
    if 'max_tokens' in config:
        tokens = config['max_tokens']
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
            
    # Set default values for optional parameters
    defaults = {
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30,
        'retry_attempts': 2
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            
    return config

def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variables"""
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'grok': 'GROK_API_KEY'
    }
    
    if provider not in key_mapping:
        return None
        
    return os.getenv(key_mapping[provider])

def get_default_model(provider: str) -> str:
    """Get default model for provider"""
    defaults = {
        'openai': 'gpt-3.5-turbo',
        'claude': 'claude-3-sonnet-20240229',
        'mistral': 'mistral-medium',
        'gemini': 'gemini-pro',
        'grok': 'grok-1'
    }
    return defaults.get(provider, '')