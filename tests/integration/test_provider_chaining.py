import pytest
from llmeasy import LLMEasy
import asyncio
import json
from tests.integration.test_utils import log_test_failure, cleanup_provider
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_chaining(api_keys):
    """Test chaining multiple providers"""
    provider_pairs = [
        ('claude', 'openai'),
        ('openai', 'claude'),
    ]
    
    for provider1, provider2 in provider_pairs:
        if not api_keys.get(f'{provider1.upper()}_API_KEY') or not api_keys.get(f'{provider2.upper()}_API_KEY'):
            continue
            
        logger.info(f"\nTesting provider chain: {provider1} -> {provider2}")
        
        llm1 = None
        llm2 = None
        try:
            llm1 = LLMEasy(provider=provider1, api_key=api_keys[f'{provider1.upper()}_API_KEY'])
            llm2 = LLMEasy(provider=provider2, api_key=api_keys[f'{provider2.upper()}_API_KEY'])
            
            # First provider generates a summary
            summary_template = {
                "key_points": ["string"],
                "main_topic": "string",
                "summary": "string"
            }
            
            logger.info(f"\nGetting summary from {provider1}...")
            summary = await llm1.query(
                prompt="Explain microservices architecture",
                system="You are a software architect. Provide a concise summary with key points.",
                template=summary_template,
                output_format='json'
            )
            
            logger.info(f"\nSummary from {provider1}: {json.dumps(summary, indent=2)}")
            
            assert isinstance(summary, dict), f"Summary from {provider1} is not a dictionary"
            assert any(key in summary for key in ["key_points", "main_topic", "summary"]), \
                f"Summary from {provider1} missing required fields"
            
            # Extract points for expansion
            points_to_expand = summary.get("key_points", [])
            if not points_to_expand and "summary" in summary:
                points_to_expand = [summary["summary"]]
            
            assert points_to_expand, f"No points to expand from {provider1}"
            
            # Add delay between requests
            await asyncio.sleep(2)
            
            # Second provider expands on the summary
            expansion_template = {
                "expanded_points": [{
                    "point": "string",
                    "explanation": "string",
                    "details": ["string"]
                }]
            }
            
            logger.info(f"\nGetting expansion from {provider2}...")
            expansion = await llm2.query(
                prompt=f"Expand on these points about microservices architecture: {json.dumps(points_to_expand)}",
                system="You are a software architect. Provide detailed explanations for each point.",
                template=expansion_template,
                output_format='json'
            )
            
            logger.info(f"\nExpansion from {provider2}: {json.dumps(expansion, indent=2)}")
            
            assert isinstance(expansion, dict), f"Expansion from {provider2} is not a dictionary"
            assert "expanded_points" in expansion, f"Expansion from {provider2} missing expanded_points"
            assert len(expansion["expanded_points"]) > 0, f"No expanded points from {provider2}"
            
            # Validate expanded points structure
            for point in expansion["expanded_points"]:
                assert isinstance(point, dict), "Expanded point is not a dictionary"
                assert any(key in point for key in ["point", "explanation", "details"]), \
                    "Expanded point missing required fields"
            
            logger.info(f"✅ Successfully tested provider chain: {provider1} -> {provider2}")
            
        except Exception as e:
            log_test_failure(
                f"{provider1}->{provider2}",
                e,
                context="During provider chaining test"
            )
            raise
            
        finally:
            # Clean up both providers
            if llm1:
                await cleanup_provider(llm1)
            if llm2:
                await cleanup_provider(llm2)
            await asyncio.sleep(2)  # Add longer delay between chain tests

@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_provider_usage(api_keys):
    """Test using multiple providers concurrently"""
    providers = ['claude', 'openai', 'mistral']
    llms = []
    
    try:
        # Create LLM instances
        for provider in providers:
            if api_key := api_keys.get(f'{provider.upper()}_API_KEY'):
                llms.append(LLMEasy(provider=provider, api_key=api_key))
        
        # Send concurrent requests
        async def get_response(llm: LLMEasy, prompt: str) -> str:
            return await llm.query(prompt, system="Be brief.")
            
        tasks = [get_response(llm, f"What is {i}?") for i, llm in enumerate(llms)]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == len(llms), "Not all concurrent requests completed"
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
            
    finally:
        for llm in llms:
            await cleanup_provider(llm)