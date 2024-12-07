from typing import Optional, Any, AsyncIterator, Union
from .providers.base import LLMProvider
from .providers.claude import ClaudeProvider
from .providers.openai import OpenAIProvider
from .templates.template_parser import PromptTemplate
from .utils.config import settings

class LLMQuery:
    """Main class for interacting with LLM providers"""
    
    def __init__(self, provider: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize LLMQuery
        
        Args:
            provider: Name of the LLM provider ('claude' or 'openai')
            api_key: Provider API key (optional, can be set in environment)
            **kwargs: Additional provider-specific configuration
        """
        self.provider = self._get_provider(provider, api_key, **kwargs)
        
    def _get_provider(self, provider: str, api_key: Optional[str], **kwargs) -> LLMProvider:
        """Get the appropriate provider instance"""
        providers = {
            'claude': (ClaudeProvider, settings.anthropic_api_key),
            'openai': (OpenAIProvider, settings.openai_api_key),
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
            
        ProviderClass, default_api_key = providers[provider]
        api_key = api_key or default_api_key
        
        if not api_key:
            raise ValueError(f"No API key provided for {provider}. Please provide an API key or set it in the environment.")
            
        return ProviderClass(api_key=api_key, **kwargs)
        
    async def query(
        self, 
        template: str, 
        variables: dict,
        output_format: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Any, AsyncIterator[str]]:
        """
        Query LLM with formatted template
        
        Args:
            template: Prompt template string
            variables: Variables to substitute in template
            output_format: Expected output format
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Formatted response from LLM or AsyncIterator for streaming
        """
        # Format template with variables
        prompt_template = PromptTemplate(template)
        formatted_prompt = prompt_template.format(**variables)
        
        # Handle streaming
        if stream:
            return await self.stream(template, variables, **kwargs)
        
        # Send query to provider
        return await self.provider.query(
            prompt=formatted_prompt,
            output_format=output_format,
            **kwargs
        )
        
    async def stream(
        self,
        template: str,
        variables: dict,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream responses from LLM with formatted template
        
        Args:
            template: Prompt template string
            variables: Variables to substitute in template
            **kwargs: Additional provider-specific parameters
            
        Returns:
            AsyncIterator yielding response chunks
        """
        # Format template with variables
        prompt_template = PromptTemplate(template)
        formatted_prompt = prompt_template.format(**variables)
        
        # Get response generator from provider
        generator = await self.provider._generate_response(
            prompt=formatted_prompt,
            stream=True,
            **kwargs
        )
        
        # Return the generator directly
        return generator
