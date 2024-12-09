import json
from typing import Any, Optional, AsyncIterator, Union
import google.generativeai as genai
from .base import LLMProvider
from llmeasy.utils import settings

class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize Gemini provider"""
        super().__init__(api_key, **kwargs)
        genai.configure(api_key=api_key)
        self.model = self.config.model or settings.gemini_model
        self.client = genai.GenerativeModel(self.model)

    async def _generate_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate response from Gemini"""
        try:
            # Combine system prompt and user prompt for Gemini
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            
            # Remove parameters that Gemini API doesn't accept
            kwargs.pop('max_tokens', None)
            kwargs.pop('output_format', None)
            kwargs.pop('system', None)
            
            response = await self.client.generate_content_async(
                full_prompt,
                generation_config={
                    'temperature': self.config.temperature,
                    'max_output_tokens': self.config.max_tokens,
                },
                stream=stream,
                **kwargs
            )
            
            if stream:
                async def response_generator():
                    async for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return response_generator()
            
            return response.text
            
        except Exception as e:
            raise ValueError(f"Error generating Gemini response: {str(e)}")

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