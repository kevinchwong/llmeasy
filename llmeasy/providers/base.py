from abc import ABC, abstractmethod
from typing import Any, Optional, AsyncIterator, Union
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
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Send query to LLM provider"""
        formatted_prompt = self._format_prompt(prompt, output_format)
        response = await self._generate_response(
            prompt=formatted_prompt,
            system=system,
            output_format=output_format,
            **kwargs
        )
        
        if not self.validate_response(response, output_format):
            raise ValueError(f"Response does not match expected {output_format} format")
            
        return self._parse_response(response, output_format)

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream responses from the LLM provider"""
        formatted_prompt = self._format_prompt(prompt, output_format)
        generator = await self._generate_response(
            prompt=formatted_prompt,
            system=system,
            stream=True,
            output_format=output_format,
            **kwargs
        )
        async for chunk in generator:
            yield chunk

    @abstractmethod
    async def _generate_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        output_format: Optional[str] = None,
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
