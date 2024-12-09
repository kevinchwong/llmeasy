from typing import Dict, Any
from string import Template

class PromptTemplate:
    def __init__(self, template: str):
        """Initialize template with string.Template"""
        self.template = Template(template)
        
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
            
        Raises:
            KeyError: If a required template variable is missing
            ValueError: If template substitution fails
        """
        try:
            # Handle None values
            cleaned_kwargs = {
                k: (str(v) if v is not None else '')
                for k, v in kwargs.items()
            }
            return self.template.substitute(**cleaned_kwargs)
        except KeyError as e:
            # Re-raise KeyError for missing variables
            raise KeyError(f"Missing required template variable: {str(e)}")
        except ValueError as e:
            # Handle other template errors
            raise ValueError(f"Template formatting error: {str(e)}")