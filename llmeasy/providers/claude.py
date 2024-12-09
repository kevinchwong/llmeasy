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
        self.model = kwargs.get('model') or settings.claude_model

    async def _generate_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate response from Claude"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
                system=system if system else "You are a helpful AI assistant.",
                stream=stream
            )
            
            if stream:
                async def response_generator():
                    async for chunk in response:
                        if hasattr(chunk, 'type'):
                            if chunk.type == 'content_block_delta':
                                if chunk.delta.text:
                                    yield chunk.delta.text
                return response_generator()
            
            return response.content[0].text
            
        except Exception as e:
            raise ValueError(f"Error generating Claude response: {str(e)}")

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream responses from Claude"""
        try:
            formatted_prompt = self._format_prompt(prompt, output_format)
            response_generator = await self._generate_response(
                prompt=formatted_prompt,
                system=system,
                stream=True,
                **kwargs
            )
            async for chunk in response_generator:
                yield chunk
        except Exception as e:
            print(f"Error in Claude stream: {str(e)}")
            raise

    async def query(
        self,
        prompt: str,
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate complete response from Claude"""
        try:
            formatted_prompt = self._format_prompt(prompt, output_format)
            result = await self._generate_response(
                prompt=formatted_prompt,
                system=system,
                stream=False,
                **kwargs
            )
            
            if output_format:
                return self._parse_response(result, output_format)
            return result
            
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