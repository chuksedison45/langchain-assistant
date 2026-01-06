# LangChain Assistant

[![Lint](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/lint.yml)
[![Tests](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/test.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

An AI application built with LangChain and AWS services for multilingual conversations and text processing.

## Features

- 🤖 Multilingual AI assistant with AWS Bedrock
- 📝 Multiple prompt templates for different tasks
- 🔄 Output parsing for clean responses
- ✅ Comprehensive test suite
- 🚀 CI/CD with GitHub Actions
- 🐍 Support for Python 3.10 and 3.11

## Project Structure

```
langchain-assistant/
├── .github/workflows/          # GitHub Actions CI/CD
│   ├── lint.yml               # Linting workflow
│   └── test.yml               # Testing workflow
├── src/                       # Source code
│   ├── __init__.py
│   ├── bedrock_client.py      # AWS Bedrock client
│   ├── chain.py              # LangChain chain builder
│   ├── config.py             # Configuration management
│   ├── main.py               # Main application
│   └── prompts.py            # Prompt templates
├── tests/                     # Test files
│   ├── __init__.py
│   ├── test_prompts.py       # Prompt template tests
│   ├── test_chain_mocked.py  # Mocked chain tests
│   └── test_integration.py   # Integration tests
├── examples/                  # Usage examples
├── .env.example              # Environment template
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── run_tests.py             # Test runner
```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/langchain-assistant.git
   cd langchain-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

5. **Test the setup**
   ```bash
   python -m pytest tests/ -v
   ```

## Usage

### Quick Start
```python
from src.main import LangChainAssistant

# Create assistant
assistant = LangChainAssistant()

# Chat in different languages
response = assistant.chat("Hello, how are you?", language="English")
print(response)

# Summarize text
summary = assistant.summarize("Long text here...", length="brief")

# Translate text
translation = assistant.translate("Hello world", source_language="English", target_language="Spanish")
```

### Command Line
```bash
# Run demo
python run.py

# Run interactive mode
python run.py --mode interactive

# Run tests
python run_tests.py
```

## Multiple Prompt Templates

The application supports multiple specialized prompts:

- **assistant**: General multilingual assistant
- **summarizer**: Text summarization with length control (brief/medium/detailed)
- **translator**: Text translation with context
- **coder**: Code generation and explanation
- **analyst**: Data analysis and insights

```python
# Switch between tasks
assistant.set_task("summarizer")
assistant.set_task("translator", target_language="German")
```

## Testing

### Run Tests Locally
```bash
# Run all tests
python run_tests.py

# Run specific test file
python -m pytest tests/test_prompts.py -v

# Run with coverage
python run_tests.py --coverage
```

### Test Coverage
The test suite includes:
- ✅ Prompt template formatting tests
- ✅ Error handling for invalid prompts
- ✅ Mocked chain tests (no AWS credentials needed)
- ✅ Integration tests
- ✅ Edge case testing

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration:

### Workflows
1. **Linting** (`lint.yml`): Runs flake8 and Black on every push
2. **Testing** (`test.yml`): Runs pytest with Python 3.10 and 3.11 matrix
3. **Full CI** (`full-ci.yml`): Complete pipeline with linting, testing, and verification

### Matrix Testing
Tests run on multiple Python versions:
- Python 3.10
- Python 3.11

### Status Badges
Add these to your README (replace YOUR_USERNAME):
```markdown
[![Lint](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/lint.yml)
[![Tests](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/langchain-assistant/actions/workflows/test.yml)
```

## Development

### Code Quality
```bash
# Run linters
flake8 src/ tests/
black --check src/ tests/

# Auto-format code
black src/ tests/
```

### Adding New Tests
Follow pytest conventions:
```python
def test_new_feature():
    """Test description."""
    # Arrange
    factory = PromptFactory()
    
    # Act
    result = factory.get_prompt_template("assistant")
    
    # Assert
    assert isinstance(result, ChatPromptTemplate)
```

## Success Criteria Checklist

### Part 1: Project Setup
- ✅ Project directory with `src/` and `tests/` subdirectories
- ✅ Virtual environment and dependencies installed
- ✅ `requirements.txt` with required packages
- ✅ `.env` file created (not committed)
- ✅ `.gitignore` includes `.env`, `__pycache__/`, `venv/`
- ✅ Repository created and pushed via CLI

### Part 2: Basic LangChain Application
- ✅ Bedrock client initializes successfully
- ✅ ChatBedrock model configured
- ✅ Prompt template with language variable
- ✅ Chain built using LCEL pipe operator (`|`)
- ✅ `chat()` function returns AI responses
- ✅ Test script demonstrates responses in 2+ languages

### Part 3: Multiple Prompt Templates
- ✅ Prompts module with at least 2 templates
- ✅ Assistant prompt with language and message variables
- ✅ Summarizer prompt with length and text variables
- ✅ `get_prompt_template()` function works correctly

### Part 4: Add Output Parsing
- ✅ `StrOutputParser` imported and added to chain
- ✅ Chain returns clean string output
- ✅ Both assistant and summarizer chains work

### Part 5: Basic Testing
- ✅ `tests/test_prompts.py` exists
- ✅ Tests verify prompt formatting works
- ✅ Tests verify prompt selector works
- ✅ Tests verify error handling for invalid prompts
- ✅ All tests pass with `pytest tests/ -v`

### Part 6: GitHub Actions CI/CD
- ✅ `.github/workflows/lint.yml` runs flake8
- ✅ `.github/workflows/test.yml` runs pytest
- ✅ Matrix strategy tests Python 3.10 and 3.11
- ✅ Both workflows pass (green checkmarks)
- ✅ README contains status badge

## Troubleshooting

### AWS Credentials Issues
```bash
# Configure AWS CLI
aws configure --profile default

# Check AWS credentials
aws sts get-caller-identity
```

### Test Failures
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run with more details
pytest tests/ -v --tb=long
```

### CI/CD Issues
- Ensure workflows are in `.github/workflows/`
- Check Python version compatibility
- Verify test files don't require AWS credentials

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.