import pytest
from llmeasy import LLMEasy
from integration.test_utils import log_test_failure, cleanup_provider
import json
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_json_handling(api_keys):
    """Test JSON handling functionality"""
    provider = 'openai'
    llm = None
    try:
        llm = LLMEasy(
            provider=provider,
            api_key=api_keys['OPENAI_API_KEY']
        )
        
        # Test JSON response with simple structure
        simple_response = await llm.query(
            prompt="Return a JSON object with your name and role",
            system="Return only valid JSON",
            output_format='json'
        )
        
        # Basic validation
        assert isinstance(simple_response, dict), "Response is not a dictionary"
        assert len(simple_response) > 0, "Empty response"
        
        # Test complex JSON response
        complex_response = await llm.query(
            prompt=(
                "Return a JSON object with the following structure:\n"
                "- users: array of user objects\n"
                "- each user should have: name, age, email, and address\n"
                "- address should have: street, city, state, zip\n"
                "Generate 2-3 example users"
            ),
            system="Return only valid JSON",
            output_format='json'
        )
        
        # Validate complex response
        assert isinstance(complex_response, dict), "Complex response is not a dictionary"
        assert 'users' in complex_response, "Missing users array"
        assert isinstance(complex_response['users'], list), "Users is not an array"
        assert len(complex_response['users']) >= 2, "Not enough users"
        
        # Validate user structure
        for user in complex_response['users']:
            assert isinstance(user, dict), "User is not an object"
            assert all(key in user for key in ['name', 'age', 'email', 'address']), "Missing user fields"
            assert isinstance(user['address'], dict), "Address is not an object"
            assert all(key in user['address'] for key in ['street', 'city', 'state', 'zip']), "Missing address fields"
        
        # Test malformed JSON handling
        malformed_prompt = "Return an incomplete JSON object"
        try:
            await llm.query(
                prompt=malformed_prompt,
                system="Return invalid JSON",
                output_format='json'
            )
        except ValueError as e:
            assert "JSON" in str(e), "Expected JSON validation error"
        
        logger.info("✅ Successfully tested JSON handling")
        
    except Exception as e:
        log_test_failure(
            provider,
            e,
            context="During JSON handling test"
        )
        raise
        
    finally:
        if llm:
            await cleanup_provider(llm) 