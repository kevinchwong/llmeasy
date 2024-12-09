"""
Core LLMEasy implementation
"""
from typing import Optional, Dict, Any, AsyncGenerator
from .providers import (
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    MistralProvider,
    GrokProvider
)
from .utils.json_helper import JSONStreamHelper
import asyncio
from .utils import settings

class LLMEasy:
    """Main LLMEasy class for managing different LLM providers"""
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        **kwargs
    ):
        """Initialize LLMEasy with specified provider"""
        self.provider = provider
        self.api_key = api_key
        
        # Get provider-specific settings
        provider_settings = settings.get_provider_settings(provider)
        common_settings = settings.get_common_settings()
        
        # Combine settings with kwargs (kwargs take precedence)
        all_settings = {**common_settings, **provider_settings, **kwargs}
        
        # Initialize the appropriate provider
        if self.provider == 'claude':
            self.provider = ClaudeProvider(api_key=api_key, **all_settings)
        elif self.provider == 'openai':
            self.provider = OpenAIProvider(api_key=api_key, **all_settings)
        elif self.provider == 'gemini':
            self.provider = GeminiProvider(api_key=api_key, **all_settings)
        elif self.provider == 'mistral':
            self.provider = MistralProvider(api_key=api_key, **all_settings)
        elif self.provider == 'grok':
            self.provider = GrokProvider(api_key=api_key, **all_settings)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the provider"""
        async for chunk in self.provider.stream(
            prompt=prompt,
            system=system,
            output_format=output_format,
            **kwargs
        ):
            yield chunk

    async def query(
        self,
        prompt: str,
        system: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Get a complete response from the provider"""
        return await self.provider.query(
            prompt=prompt,
            system=system,
            output_format=output_format,
            **kwargs
        ) 

    async def stream_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[Dict] = None,
        validator = None,
        **kwargs
    ) -> AsyncGenerator[Dict[Any, Any], None]:
        """Stream responses as JSON objects"""
        
        self.json_helper = JSONStreamHelper(template)
        
        stream = self.stream(
            prompt=prompt,
            system=system,
            output_format='json',
            **kwargs
        )
        
        async for json_obj in self.json_helper.process_stream(
            stream,
            validator
        ):
            yield json_obj

    async def batch_process(
        self,
        prompts: list[str],
        system: Optional[str] = None,
        max_concurrent: int = 3,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Process multiple prompts concurrently"""
        
        async def process_prompt(prompt):
            async for response in self.stream(prompt, system, **kwargs):
                yield response
                
        tasks = [process_prompt(prompt) for prompt in prompts]
        
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            for response in await asyncio.gather(*batch):
                yield response 