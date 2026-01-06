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

## Multiple Prompt Templates

The application now supports multiple specialized prompt templates:

### Available Tasks

1. **assistant** - General multilingual assistant
   ```python
   assistant.chat("Your message", language="Spanish")
   ```

2. **summarizer** - Text summarization with length control
   ```python
   assistant.summarize(text, length="brief")  # or "medium", "detailed"
   ```

3. **translator** - Text translation with context
   ```python
   assistant.translate(text, source_language="English", target_language="French", context="Formal")
   ```

4. **coder** - Code generation and explanation
   ```python
   assistant.code(request, language="Python", task_type="implementation")
   ```

5. **analyst** - Data analysis and insights
   ```python
   assistant.analyze(data, question, focus="business", audience="executives")
   ```

### Usage Examples

```bash
# Run the multiple prompt demo
python -c "from src.main import demo_multiple_tasks; demo_multiple_tasks()"

# Run comprehensive tests
python scripts/test_prompts.py

# See example usage
python examples/example_usage.py
```

### Switching Between Tasks

```python
from src.main import LangChainAssistant

assistant = LangChainAssistant()

# Switch tasks
assistant.set_task("summarizer")
assistant.set_task("translator", target_language="German")

# Get task information
info = assistant.get_task_info("coder")
print(f"Required inputs: {info['required_inputs']}")
```

## Testing

Test all prompt templates:
```bash
python scripts/test_prompts.py
```

## Part 4: Output Parsing

The application now includes advanced output parsing for clean, structured responses.

### Output Parsers Available

1. **clean_text** - Clean string output with formatting
2. **clean_summary** - Formatted summaries with proper structure
3. **json** - JSON output for structured data
4. **structured** - Pydantic model-based structured output

### Usage Examples

```python
from src.main import LangChainAssistant

assistant = LangChainAssistant()

# Basic chat with clean text output
response = assistant.chat("Your question", "English", output_parser="clean_text")

# Summarizer with clean summary output
summary = assistant.summarize(text, length="brief", output_parser="clean_summary")

# Structured JSON output
structured_response = assistant.analyze_structured("Analyze this topic", "English")

## Part 5: Basic Testing

The project includes a comprehensive test suite for prompt templates and functionality.

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py --file tests/test_prompts.py

# Run tests with coverage report
python run_tests.py --coverage

# Directly with pytest
pytest tests/ -v

# Run only prompt tests
pytest tests/test_prompts.py -v

# Run mocked chain tests
pytest tests/test_chain_mocked.py -v

```

## Test Structure
```text
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_prompts.py          # Tests for prompt templates and factory
├── test_chain_mocked.py     # Mocked tests for chain functionality
└── test_integration.py      # Integration tests
```

## Test Requirements
- Tests run without AWS credentials

- Only test prompt formatting and logic

- Use mocking for AWS/Bedrock dependencies

## Test Coverage
The test suite covers:

1. **Prompt Factory Tests**

    - Prompt template creation for all tasks

    - Input variable validation

    - Error handling for invalid tasks

    - Prompt formatting with sample inputs

2. **Prompt Selector Tests**

    - get_prompt_by_name() returns correct template

    - Case-insensitive task names

    - Proper error messages for unknown tasks

3. **Integration Tests**

    - Components work together correctly

    - LangChain compatibility

    - Error handling flows

### Writing New Tests
Follow pytest conventions:

```python
def test_feature_description():
    """Test description."""
    # Arrange
    factory = PromptFactory()
    
    # Act
    result = factory.get_prompt_template("assistant")
    
    # Assert
    assert isinstance(result, ChatPromptTemplate)
```

## Mocking AWS Dependencies
For tests that involve AWS, use mocking:

```python
from unittest.mock import Mock, patch

@patch('boto3.Session')
@patch('langchain_aws.ChatBedrock')
def test_chain_with_mocked_aws(mock_chat, mock_session):
    # Test chain creation without real AWS calls
    pass

```