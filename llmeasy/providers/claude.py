import json
from typing import Any, Optional, AsyncIterator, Union
import anthropic
from .base import LLMProvider
from llmeasy.utils import settings

class ClaudeProvider(LLMProvider):
    """Provider for Anthropic's Claude models"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize Claude provider"""
        super().__init__(api_key, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = self.config.model or settings.claude_model

    async def _generate_response(
        self,
        prompt: str,
        stream: bool = False,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate response from Claude"""
        try:
            # Remove parameters that Claude API doesn't accept
            kwargs.pop('max_tokens', None)
            kwargs.pop('output_format', None)
            
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=stream,
                system="You are a helpful AI assistant. Always provide responses in the requested format.",
                **kwargs
            )
            
            if stream:
                async def response_generator():
                    async for chunk in message:
                        if chunk.delta.text:
                            yield chunk.delta.text
                return response_generator()
            
            return message.content[0].text
            
        except Exception as e:
            raise ValueError(f"Error generating Claude response: {str(e)}")

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