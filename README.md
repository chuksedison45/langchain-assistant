# LangChain Assistant

An AI application built with LangChain and AWS services.

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Configure AWS credentials in `.env` file

## Project Structure

- `src/` - Source code
- `tests/` - Test files
- `docs/` - Documentation

## Usage

### Quick Start
```bash
# Run the demo (pre-configured conversations)
python run.py

# Run interactive chat
python run.py --mode interactive

# Run tests
python run.py --mode test
```

## Simple Test
```bash
python test_simple.py
```

## Advanced Usage
```python
from src.main import LangChainAssistant

# Create assistant
assistant = LangChainAssistant()

# Single query
response = assistant.chat(
    "Your message here",
    language="Spanish"
)

# Batch processing
conversations = [
    {"message": "Hello", "language": "English"},
    {"message": "Hola", "language": "Spanish"},
]
results = assistant.batch_chat(conversations)
```
## Features
- Multilingual support (any language the model supports)

- Configurable AI models via AWS Bedrock

- Conversation history tracking

- Batch processing capabilities

- Error handling and retries


