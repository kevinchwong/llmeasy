import pytest
from llmeasy.templates.template_parser import PromptTemplate

def test_template_substitution():
    template = PromptTemplate("Hello, ${name}!")
    result = template.format(name="World")
    assert result == "Hello, World!"

def test_missing_variable():
    template = PromptTemplate("Hello, ${name}!")
    with pytest.raises(ValueError):
        template.format(wrong_name="World")

def test_multiple_variables():
    template = PromptTemplate("${greeting}, ${name}! How is the ${time_of_day}?")
    result = template.format(
        greeting="Hello",
        name="Alice",
        time_of_day="morning"
    )
    assert result == "Hello, Alice! How is the morning?"

def test_empty_template():
    template = PromptTemplate("")
    result = template.format()
    assert result == "" 