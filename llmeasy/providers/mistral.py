import json
from typing import Any, Optional, AsyncIterator, Union
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from .base import LLMProvider
from llmeasy.utils import settings

class MistralProvider(LLMProvider):
    """Provider for Mistral AI models"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize Mistral provider"""
        super().__init__(api_key, **kwargs)
        self.client = MistralClient(api_key=api_key)
        self.model = self.config.model or settings.mistral_model

    async def _generate_response(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate response from Mistral"""
        try:
            # Remove parameters that Mistral API doesn't accept
            kwargs.pop('max_tokens', None)
            kwargs.pop('output_format', None)
            
            messages = [ChatMessage(role="user", content=prompt)]
            
            response = await self.client.chat_async(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                async def response_generator():
                    async for chunk in response:
                        if chunk.delta:
                            yield chunk.delta
                return response_generator()
            
            return response.content
            
        except Exception as e:
            raise ValueError(f"Error generating Mistral response: {str(e)}")

    def _format_prompt(self, prompt: str, output_format: Optional[str]) -> str:
        """Format prompt with output instructions"""
        if output_format == 'json':
            prompt += "\nProvide your response in valid JSON format only. Do not include any explanatory text outside the JSON structure."
        elif output_format:
            prompt += f"\nProvide your response in {output_format} format."
        return prompt

    def _parse_response(self, response: str, output_format: Optional[str]) -> Any:
        """Parse and validate response"""
        if output_format == 'json':
            try:
                # Strip any potential markdown formatting
                json_str = response.strip().strip('```json').strip('```').strip()
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response: {str(e)}\nResponse: {response}")
        return response

    def validate_response(self, response: str, output_format: Optional[str]) -> bool:
        """Validate response format"""
        if output_format == 'json':
            try:
                # Strip any potential markdown formatting
                json_str = response.strip().strip('```json').strip('```').strip()
                json.loads(json_str)
                return True
            except json.JSONDecodeError:
                return False
        return True

