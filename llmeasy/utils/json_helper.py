import json
import re
from typing import AsyncGenerator, Dict, Any, Optional, Union, List
from collections import defaultdict

class JSONStreamHelper:
    """Enhanced helper class for robust JSON streaming scenarios"""
    
    def __init__(self, template: Optional[Dict] = None):
        self.template = template
        self.buffer = ""
        self.depth_counter = defaultdict(int)
        
    def _find_json_boundaries(self, text: str) -> List[tuple[int, int]]:
        """Find potential JSON object boundaries in text"""
        stack = []
        boundaries = []
        in_string = False
        escape_char = False
        start_index = None
        
        for i, char in enumerate(text):
            # Handle string literals
            if char == '"' and not escape_char:
                in_string = not in_string
            elif char == '\\' and not escape_char:
                escape_char = True
                continue
            
            if not in_string:
                if char == '{':
                    if not stack:
                        start_index = i
                    stack.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack:  # Complete object found
                            boundaries.append((start_index, i + 1))
            
            escape_char = False
            
        return boundaries
    
    def _clean_json_text(self, text: str) -> str:
        """Clean and normalize JSON text"""
        # Remove common formatting issues
        text = re.sub(r'[\n\r\t]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Fix common JSON syntax errors
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)
        # Ensure proper quote usage
        text = re.sub(r'(?<!\\)"', '"', text)  # Fix quotes
        return text.strip()
        
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON"""
        # Balance braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            text = '{' * (close_braces - open_braces) + text
        
        # Ensure property names are quoted
        text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)
        
        return text
    
    async def process_stream(
        self, 
        stream: AsyncGenerator[str, None],
        validator = None,
        repair_json: bool = True
    ) -> AsyncGenerator[Dict[Any, Any], None]:
        """Process a stream of text into JSON objects with enhanced error handling"""
        
        async for chunk in stream:
            self.buffer += chunk
            self.buffer = self._clean_json_text(self.buffer)
            
            while True:  # Keep trying to process buffer until no valid JSON is found
                # Find potential JSON objects
                boundaries = self._find_json_boundaries(self.buffer)
                
                if not boundaries:
                    break  # No complete JSON objects found, wait for more data
                    
                found_valid = False
                for start, end in boundaries:
                    json_str = self.buffer[start:end]
                    
                    try:
                        if repair_json:
                            json_str = self._repair_json(json_str)
                        
                        json_obj = json.loads(json_str)
                        
                        # Validate object
                        if not isinstance(json_obj, dict):
                            continue
                            
                        if validator and not validator(json_obj):
                            continue
                        
                        yield json_obj
                        self.buffer = self.buffer[end:].lstrip()
                        found_valid = True
                        break
                        
                    except json.JSONDecodeError as e:
                        if repair_json:
                            # Try one more time with aggressive repair
                            try:
                                json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                                json_str = self._repair_json(json_str)
                                json_obj = json.loads(json_str)
                                
                                if isinstance(json_obj, dict) and \
                                   (not validator or validator(json_obj)):
                                    yield json_obj
                                    self.buffer = self.buffer[end:].lstrip()
                                    found_valid = True
                                    break
                            except:
                                continue
                        continue
                    except Exception:
                        continue
                
                if not found_valid:
                    break  # No valid JSON found in any boundary, wait for more data
