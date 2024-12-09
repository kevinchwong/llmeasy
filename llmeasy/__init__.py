"""
LLMEasy - A flexible framework for integrating multiple AI chat providers
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "Apache License 2.0"

from .core import LLMEasy
from .providers import *  # noqa

__all__ = [
    "LLMEasy",
    "__version__",
    "__author__",
    "__license__",
]
