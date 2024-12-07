from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator, Union
from pydantic import BaseModel, ConfigDict

class ProviderConfig(BaseModel):
    """Configuration for LLM providers"""
    model_config = ConfigDict(
        extra='allow',
        json_schema_extra={
            'env': True
        }
    )
    
    api_key: str
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize provider with API key and configuration"""
        self.config = ProviderConfig(api_key=api_key, **kwargs)

    async def query(
        self, 
        prompt: str, 
        output_format: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Send query to LLM provider
        
        Args:
            prompt: The formatted prompt to send
            output_format: Expected output format (json, string, etc)
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response from the LLM provider
        """
        # Format prompt with output instructions
        formatted_prompt = self._format_prompt(prompt, output_format)
        
        # Get response from provider
        response = await self._generate_response(
            formatted_prompt,
            stream=stream,
            output_format=output_format,
            **kwargs
        )
        
        # Handle streaming response
        if stream:
            return response
            
        # Validate and parse response
        if not self.validate_response(response, output_format):
            raise ValueError(f"Response does not match expected {output_format} format")
            
        return self._parse_response(response, output_format)

    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream responses from the LLM provider"""
        return await self._generate_response(prompt, stream=True, **kwargs)

    @abstractmethod
    async def _generate_response(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Internal method to generate responses"""
        pass

    @abstractmethod
    def _format_prompt(self, prompt: str, output_format: Optional[str]) -> str:
        """Format the prompt with output format instructions"""
        pass

    @abstractmethod
    def _parse_response(self, response: Any, output_format: Optional[str]) -> Any:
        """Parse and validate the response"""
        pass

    @abstractmethod
    def validate_response(self, response: Any, output_format: Optional[str]) -> bool:
        """Validate if response matches expected format"""
        pass
