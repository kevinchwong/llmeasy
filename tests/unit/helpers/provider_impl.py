from llmeasy.base import BaseProvider
from typing import Optional, Any
import json

class MockProvider(BaseProvider):
    """Mock implementation of BaseProvider for testing"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = kwargs.get('api_key')
        
    async def query(self, prompt: str, **kwargs):
        return "Test response"
        
    async def stream(self, prompt: str, **kwargs):
        yield "Test chunk"
        
    def _format_prompt(self, prompt: str, output_format: Optional[str] = None) -> str:
        """Override format prompt to handle templates"""
        if output_format == 'json':
            return f"{prompt}\nProvide response in JSON format."
        return prompt
        
    def _parse_response(self, response: str, output_format: Optional[str]) -> Any:
        """Override parse response"""
        if output_format == 'json':
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return response
        return response
        
    def validate_response(self, response: str, output_format: Optional[str]) -> bool:
        """Override validate response"""
        if output_format == 'json':
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        return True 