from typing import Optional
from pydantic import BaseModel, ConfigDict

class Settings(BaseModel):
    """Configuration settings for LLM providers"""
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    
    # Model configurations
    claude_model: str = "claude-3-sonnet-20240229"
    openai_model: str = "gpt-4-turbo-preview"
    gemini_model: str = "gemini-pro"
    mistral_model: str = "mistral-large-latest"
    grok_model: str = "grok-beta"
    
    # Common Configuration
    max_tokens: int = 1000
    temperature: float = 0.7

# Create and export settings instance
settings = Settings()

# Export the Settings class and settings instance
__all__ = ['Settings', 'settings']