import json
from typing import Any, Optional, AsyncIterator, Union
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from ..base import BaseProvider
from typing import AsyncGenerator, Dict, Any
import asyncio
from functools import partial

class MistralProvider(BaseProvider):
    """Provider for Mistral AI models"""
    
    # Supported Mistral parameters
    SUPPORTED_PARAMS = {
        'temperature',
        'top_p',
        'max_tokens',
        'safe_mode',
        'random_seed'
    }
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize Mistral provider"""
        super().__init__(**kwargs)
        self.client = MistralClient(api_key=api_key)
        self.model = kwargs.get('model', 'mistral-medium')

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only include supported parameters"""
        return {k: v for k, v in kwargs.items() if k in self.SUPPORTED_PARAMS}

    async def query(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))

        # Filter kwargs to only include supported parameters
        filtered_kwargs = self._filter_kwargs(kwargs)

        # Run the synchronous chat method in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                self.client.chat,
                model=self.model,
                messages=messages,
                **filtered_kwargs
            )
        )
        
        # Extract content from the correct response structure
        try:
            # First try the new response structure
            return response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            # Fallback to older response structure if available
            try:
                return response.choices[0].delta.content
            except (AttributeError, IndexError):
                raise ValueError(f"Unable to extract content from Mistral response: {str(e)}\nResponse structure: {response}")

    async def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))

        # Filter kwargs to only include supported parameters
        filtered_kwargs = self._filter_kwargs(kwargs)

        # Get the synchronous stream
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            partial(
                self.client.chat_stream,
                model=self.model,
                messages=messages,
                **filtered_kwargs
            )
        )

        # Process the stream in chunks
        try:
            for chunk in stream:
                try:
                    # Try new response structure
                    if content := chunk.choices[0].delta.content:
                        yield content
                except (AttributeError, IndexError):
                    # Try alternative response structures
                    try:
                        if content := chunk.choices[0].message.content:
                            yield content
                    except (AttributeError, IndexError) as e:
                        logger.warning(f"Unable to extract content from chunk: {e}")
                        continue
        except Exception as e:
            raise ValueError(f"Error processing Mistral stream: {str(e)}")

    def _format_prompt(self, prompt: str, output_format: Optional[str]) -> str:
        """Format prompt with output instructions"""
        formatted_prompt = prompt
        if output_format == 'json':
            formatted_prompt += "\nProvide your response in valid JSON format only. Do not include any explanatory text outside the JSON structure."
        elif output_format:
            formatted_prompt += f"\nProvide your response in {output_format} format."
        return formatted_prompt

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

