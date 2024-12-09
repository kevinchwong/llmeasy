from typing import Optional, AsyncGenerator, Any, Dict
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, **kwargs):
        """Initialize base provider with common settings"""
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.top_p = kwargs.get('top_p', None)
        self.frequency_penalty = kwargs.get('frequency_penalty', None)
        self.presence_penalty = kwargs.get('presence_penalty', None)
        
    @abstractmethod
    async def query(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Send a query to the LLM and get a response"""
        pass
        
    @abstractmethod
    async def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream responses from the LLM"""
        pass
        
    def _format_prompt(self, prompt: str, output_format: Optional[str]) -> str:
        """Format prompt with output instructions"""
        if output_format == 'json':
            prompt += "\nProvide your response in valid JSON format only. Do not include any explanatory text outside the JSON structure."
        elif output_format:
            prompt += f"\nProvide your response in {output_format} format."
        return prompt
        
    def _parse_response(self, response: str, output_format: Optional[str]) -> Any:
        """Parse and validate response format"""
        return response
        
    def validate_response(self, response: str, output_format: Optional[str]) -> bool:
        """Validate response format"""
        return True
        
    async def cleanup(self):
        """Cleanup any resources"""
        if hasattr(self, 'client') and hasattr(self.client, 'aclose'):
            await self.client.aclose() 