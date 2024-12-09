import json
from typing import Any, Optional, AsyncIterator, Union
from openai import AsyncOpenAI
from .base import LLMProvider
from llmeasy.utils import settings

class GrokProvider(LLMProvider):
    """Provider for xAI's Grok models"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize Grok provider"""
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model = self.config.model or settings.grok_model

    async def _generate_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate response from Grok"""
        try:
            # Create messages array with system prompt if provided
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            else:
                messages.append({
                    "role": "system",
                    "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
                })
            messages.append({"role": "user", "content": prompt})
            
            # Remove parameters that Grok API doesn't accept
            kwargs.pop('max_tokens', None)
            kwargs.pop('output_format', None)
            kwargs.pop('system', None)
            
            completion_kwargs = {
                'model': self.model,
                'messages': messages,
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'stream': stream,
                **kwargs
            }
            
            try:
                response = await self.client.chat.completions.create(**completion_kwargs)
            except Exception as e:
                if "model grok-1 does not exist" in str(e):
                    # Try with grok-beta if grok-1 fails
                    completion_kwargs['model'] = "grok-beta"
                    response = await self.client.chat.completions.create(**completion_kwargs)
                else:
                    raise
            
            if stream:
                async def response_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return response_generator()
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise ValueError(f"Error generating Grok response: {str(e)}")

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