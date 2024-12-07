"""
Example implementations demonstrating LLMEasy usage

This package contains various examples showing how to:
- Use different LLM providers
- Implement streaming responses
- Chain multiple providers
- Use custom templates
- Handle advanced use cases
"""

from . import templates
from .run_all_examples import main

__all__ = ["templates", "main"]
