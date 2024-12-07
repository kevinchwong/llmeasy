# AI Chat Framework

A flexible framework for integrating multiple AI chat providers (OpenAI, Claude, Gemini, Mistral, Grok).

## Installation

```bash
pip install llmeasy
```

## Setup

1. Copy `.env.example` to `.env`
2. Add your API keys to `.env`
3. Install dependencies: `poetry install`

## Environment Variables

### API Keys
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY` 
- `GOOGLE_API_KEY`
- `MISTRAL_API_KEY`
- `GROK_API_KEY`

### Models
- `CLAUDE_MODEL` (default: claude-3-sonnet-20240229)
- `OPENAI_MODEL` (default: gpt-4-turbo-preview)
- `GEMINI_MODEL` (default: gemini-pro)
- `MISTRAL_MODEL` (default: mistral-large-latest)
- `GROK_MODEL` (default: grok-beta)

### Config
- `MAX_TOKENS` (default: 1000)
- `TEMPERATURE` (default: 0.7)

## Examples

Run all examples:
```bash
python examples/run_all_examples.py
```

Available examples in `examples/`:
- Basic usage
- Provider-specific implementations
- Advanced patterns
- Custom templates
- Provider comparisons
- Streaming
- Provider chaining

## Features

- Multi-provider support
- Async/await
- Streaming responses
- Custom templates
- Provider chaining
- Error handling
- Type hints

## License

Apache License 2.0