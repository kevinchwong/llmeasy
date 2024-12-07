from typing import Dict, Any
from string import Template

class PromptTemplate:
    def __init__(self, template: str):
        self.template = Template(template)
        
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.template.substitute(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {str(e)}") 