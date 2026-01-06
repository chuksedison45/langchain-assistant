#!/usr/bin/env python3
"""
Minimal test file for pytest.
"""

import sys
import os
from prompts import PromptFactory
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_prompt_factory_exists():
    """Test that PromptFactory exists."""
    factory = PromptFactory()
    assert factory is not None


def test_get_assistant_prompt():
    """Test getting assistant prompt."""
    prompt = PromptFactory.get_prompt_template("assistant")
    assert prompt is not None

    # Test formatting
    formatted = prompt.format(language="English", message="test")
    assert "test" in formatted
    assert "English" in formatted


def test_get_summarizer_prompt():
    """Test getting summarizer prompt."""
    prompt = PromptFactory.get_prompt_template("summarizer")
    assert prompt is not None

    # Test formatting
    formatted = prompt.format(text="sample text", length="brief")
    assert "sample text" in formatted


def test_invalid_prompt():
    """Test invalid prompt raises error."""
    import pytest

    with pytest.raises(ValueError):
        PromptFactory.get_prompt_template("invalid_task")
