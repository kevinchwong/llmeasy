from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Claude Configuration
    claude_model: str = Field("claude-3-sonnet-20240229", env="CLAUDE_MODEL")
    
    # OpenAI Configuration
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Common Configuration
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings() 