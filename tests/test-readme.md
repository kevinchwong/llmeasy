# Running Integration Tests for LLMEasy

## Prerequisites

- Python 3.8 or higher
- Poetry installed
- API keys for the following services:
  - OpenAI (OPENAI_API_KEY)
  - Anthropic (ANTHROPIC_API_KEY)
  - Mistral (MISTRAL_API_KEY)
  - Google (GOOGLE_API_KEY)
  - Grok (GROK_API_KEY)

## Setup

1. Clone the repository
2. Create a `.env` file in the project root
3. Add your API keys to the `.env` file:


```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GROK_API_KEY=your_key_here
```

## Running Tests:

Run all integration tests:
```
poetry run pytest -v --log-cli-level=INFO tests/integration/ 
```

Run specific test files:
```
poetry run pytest tests/integration/test_basic_queries.py -v
poetry run pytest tests/integration/test_streaming.py -v
poetry run pytest tests/integration/test_json_templates.py -v
poetry run pytest tests/integration/test_provider_chaining.py -v
poetry run pytest tests/integration/test_error_handling.py -v
poetry run pytest tests/integration/test_model_config.py -v
```

Run a specific test:
```
poetry run pytest tests/integration/test_basic_queries.py::test_basic_query -v
```

Run with detailed logging:
```
poetry run pytest tests/integration/ -v --log-cli-level=INFO
```

Run with coverage:
```
poetry run pytest tests/integration/ -v --cov=llmeasy --cov-report=term-missing
```

Skip specific providers:
Remove or leave empty the corresponding API key in .env

## Test Categories:

Basic Queries: Tests simple queries across all providers
Streaming: Tests streaming responses
JSON Templates: Tests structured JSON outputs
Provider Chaining: Tests using multiple providers in sequence
Error Handling: Tests error cases and recovery
Model Configuration: Tests different models and settings

## Troubleshooting:

If tests fail due to rate limits, increase delay between tests
Check API keys are valid and have sufficient credits
For timeout errors, adjust timeout values in test configurations
Look for detailed error logs in pytest output

## Notes:

- Tests use retry logic for reliability
- Some providers may be skipped if API keys aren't provided
- Expect longer run times for full test suite
- Rate limits may affect test reliability
- Some providers may have known issues or limitations

## For Development:

- Add new test files in tests/integration/
- Use the existing utilities in test_utils.py
- Follow the existing patterns for consistency
- Add proper logging and error handling
- Use the retry decorator for flaky tests
