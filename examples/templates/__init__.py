"""
Custom prompt templates for different use cases
"""

from .advanced_templates import *
from .grok_templates import *
from .mistral_templates import *
from .openai_templates import *
from .comparison_templates import *
from .claude_templates import *
from .gemini_templates import *

__all__ = [
    # Advanced templates
    "AdvancedTemplate",
    "ChainedTemplate",
    "StreamingTemplate",
    
    # Grok templates
    "GrokBasicTemplate",
    "GrokAdvancedTemplate",
    
    # Mistral templates
    "MistralBasicTemplate",
    "MistralAdvancedTemplate",
    
    # OpenAI templates
    "OpenAIBasicTemplate",
    "OpenAIAdvancedTemplate",

    # Comparison templates
    "CODE_ANALYSIS_TEMPLATE",
    "TECHNICAL_EXPLANATION_TEMPLATE",
    "SUMMARY_TEMPLATE",
    "EXPANSION_TEMPLATE",

    # Claude templates
    "BASIC_EXPLANATION_TEMPLATE",
    "CODE_ANALYSIS_TEMPLATE",
    "CREATIVE_WRITING_TEMPLATE",
    "DATA_EXTRACTION_TEMPLATE",

    # Gemini templates
    "ANALYSIS_TEMPLATE",
    "CREATIVE_TEMPLATE",
    "TECHNICAL_TEMPLATE",

    # Other templates
    "CUSTOM_TEMPLATE",
]
