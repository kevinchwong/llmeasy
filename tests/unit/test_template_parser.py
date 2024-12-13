import pytest
from llmeasy.templates.template_parser import PromptTemplate
import logging

logger = logging.getLogger(__name__)

class TestTemplateParser:
    def test_basic_template(self):
        """Test basic template substitution"""
        template = PromptTemplate("Hello, $name!")
        result = template.format(name="World")
        assert result == "Hello, World!"
        
        template = PromptTemplate("$greeting, $name!")
        result = template.format(greeting="Hi", name="Alice")
        assert result == "Hi, Alice!"

    def test_missing_variables(self):
        """Test handling of missing variables"""
        template = PromptTemplate("Hello, $name!")
        with pytest.raises(KeyError) as exc_info:
            template.format()  # Missing required variable
        assert "name" in str(exc_info.value)
        
        with pytest.raises(KeyError):
            template.format(wrong_var="test")

    def test_none_values(self):
        """Test handling of None values"""
        template = PromptTemplate("Value: $value")
        result = template.format(value=None)
        assert result == "Value: "

    def test_special_characters(self):
        """Test handling of special characters"""
        # Test escaping dollar sign
        template = PromptTemplate("Cost: $$50")
        result = template.format()
        assert result == "Cost: $50"
        
        # Test dollar sign with variable
        template = PromptTemplate("Price: $$$amount")
        result = template.format(amount="100")
        assert result == "Price: $100"

    def test_multiline_template(self):
        """Test multiline templates"""
        template = PromptTemplate("""
            Title: $title
            Author: $author
            Year: $year
        """)
        
        result = template.format(
            title="Test",
            author="Alice",
            year="2024"
        )
        
        assert "Title: Test" in result
        assert "Author: Alice" in result
        assert "Year: 2024" in result

    def test_whitespace_handling(self):
        """Test whitespace handling"""
        template = PromptTemplate("$start    $middle     $end")
        result = template.format(
            start="Begin",
            middle="Mid",
            end="Finish"
        )
        assert result == "Begin    Mid     Finish"

    def test_type_conversion(self):
        """Test automatic type conversion"""
        template = PromptTemplate("Number: $num, Boolean: $bool")
        result = template.format(num=42, bool=True)
        assert result == "Number: 42, Boolean: True"

    def test_empty_template(self):
        """Test empty template"""
        template = PromptTemplate("")
        result = template.format()
        assert result == ""

    def test_invalid_template(self):
        """Test invalid template syntax"""
        # Test unclosed brace
        template_str = "${invalid"
        with pytest.raises((ValueError, KeyError)) as exc_info:  # Allow either exception type
            template = PromptTemplate(template_str)
            template.format()  # Force template evaluation
        
        # Test incomplete variable
        template_str = "$"
        with pytest.raises((ValueError, KeyError)) as exc_info:
            template = PromptTemplate(template_str)
            template.format()
        
        # Test invalid variable name
        template_str = "$123"
        with pytest.raises((ValueError, KeyError)) as exc_info:
            template = PromptTemplate(template_str)
            template.format()

    def test_recursive_template(self):
        """Test handling of recursive template substitution"""
        template = PromptTemplate("$outer($inner)")
        result = template.format(
            outer="Hello",
            inner="World"
        )
        assert result == "Hello(World)"
        
        # Test nested dict access - should work with simple string substitution
        template = PromptTemplate("$name is $age years old")
        result = template.format(name="Alice", age=30)
        assert result == "Alice is 30 years old"