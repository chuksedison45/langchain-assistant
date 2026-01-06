#!/usr/bin/env python3
"""
Pytest configuration file for the test suite.
This file provides fixtures and configuration for all tests.
"""

import pytest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fixtures that can be used across test files
@pytest.fixture
def mock_bedrock_client():
    """Fixture to provide a mocked Bedrock client."""
    from unittest.mock import Mock
    mock_client = Mock()
    mock_model = Mock()
    mock_client.create_chat_model.return_value = mock_model
    return mock_client

@pytest.fixture
def prompt_factory():
    """Fixture to provide a PromptFactory instance."""
    from prompts import PromptFactory
    return PromptFactory()

@pytest.fixture
def sample_inputs():
    """Fixture providing sample inputs for prompt testing."""
    return {
        "assistant": {
            "language": "English",
            "message": "Explain artificial intelligence in simple terms."
        },
        "summarizer": {
            "text": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks that normally require human intelligence.",
            "length": "brief"
        },
        "translator": {
            "text": "Hello, how are you today?",
            "source_language": "English",
            "target_language": "Spanish",
            "context": "Casual conversation"
        },
        "coder": {
            "message": "Write a function that calculates factorial",
            "language": "Python",
            "task_type": "implementation",
            "requirements": "Use recursion and include error handling"
        }
    }