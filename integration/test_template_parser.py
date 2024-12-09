import pytest
from llmeasy.templates.template_parser import PromptTemplate
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
def test_template_parser_basic():
    """Test basic template parsing functionality"""
    # Test simple template
    template = PromptTemplate("Hello, $name!")
    result = template.format(name="World")
    assert result == "Hello, World!"
    
    # Test multiple variables
    template = PromptTemplate("$greeting, $name! How is the $time_of_day?")
    result = template.format(
        greeting="Hi",
        name="Alice",
        time_of_day="morning"
    )
    assert result == "Hi, Alice! How is the morning?"

@pytest.mark.integration
def test_template_parser_missing_variables():
    """Test handling of missing template variables"""
    template = PromptTemplate("Hello, $name!")
    
    # Test with missing required variable
    with pytest.raises(KeyError) as exc_info:  # Changed from ValueError to KeyError
        template.format()  # Missing required variable
    assert "name" in str(exc_info.value)

@pytest.mark.integration
def test_template_parser_complex():
    """Test complex template structures"""
    template = PromptTemplate("""
    Task: $task_type
    
    Context:
    $context
    
    Instructions:
    1. $instruction1
    2. $instruction2
    
    Additional Notes:
    $notes
    """)
    
    result = template.format(
        task_type="Code Review",
        context="Python function that needs review",
        instruction1="Check for PEP8 compliance",
        instruction2="Verify error handling",
        notes="Focus on readability"
    )
    
    assert "Task: Code Review" in result
    assert "Python function that needs review" in result
    assert "Check for PEP8 compliance" in result
    assert "Verify error handling" in result
    assert "Focus on readability" in result

@pytest.mark.integration
def test_template_parser_empty_values():
    """Test handling of empty values"""
    template = PromptTemplate("Name: $name, Role: $role")
    
    # Empty strings should be allowed
    result = template.format(name="", role="")
    assert result == "Name: , Role: "

@pytest.mark.integration
def test_template_parser_special_characters():
    """Test handling of special characters in templates"""
    # Test literal dollar sign
    template = PromptTemplate("Price: $$50")  # $$ for literal $
    result = template.format()  # No variables needed
    assert result == "Price: $50"
    
    # Test dollar sign with variable
    template = PromptTemplate("Amount: $$$amount")  # First $$ for literal $, then $amount
    result = template.format(amount="100")
    assert result == "Amount: $100"

@pytest.mark.integration
def test_template_parser_whitespace():
    """Test handling of whitespace in templates"""
    template = PromptTemplate("""
        $var1
        $var2
            $var3
    """)
    
    result = template.format(
        var1="First",
        var2="Second",
        var3="Third"
    )
    
    assert "First" in result
    assert "Second" in result
    assert "Third" in result
    assert result.count("\n") >= 3  # Preserves newlines

@pytest.mark.integration
def test_template_parser_reuse():
    """Test reusing the same template with different values"""
    template = PromptTemplate("$action $object")
    
    result1 = template.format(action="Create", object="file")
    assert result1 == "Create file"
    
    result2 = template.format(action="Delete", object="folder")
    assert result2 == "Delete folder"
    
    result3 = template.format(action="Update", object="database")
    assert result3 == "Update database" 