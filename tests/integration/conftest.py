import pytest
import asyncio
import contextlib
from httpx import AsyncClient
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    with contextlib.suppress(Exception):
        loop.close()

@pytest.fixture(scope="session")
async def http_client():
    """Shared HTTP client for tests"""
    async with AsyncClient() as client:
        yield client

@pytest.fixture(scope="session")
def api_keys():
    """Fixture to provide API keys"""
    keys = {
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
        'GROK_API_KEY': os.getenv('GROK_API_KEY')
    }
    
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        pytest.skip(f"Skipping test due to missing API keys: {', '.join(missing_keys)}")
        
    return keys 