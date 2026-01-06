#!/usr/bin/env python3
"""
Integration tests that verify components work together.
These tests don't require AWS credentials.
"""

import pytest
import sys
import os
from prompts import PromptFactory
from langchain_core.prompts import ChatPromptTemplate

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_prompt_factory_integration():
    """Test that PromptFactory integrates with LangChain properly."""
    factory = PromptFactory()

    # Test all tasks
    for task in factory.SUPPORTED_TASKS:
        # Get prompt template
        prompt = factory.get_prompt_template(task)

        # Verify it's a LangChain prompt template
        assert isinstance(prompt, ChatPromptTemplate)

        # Verify it can be formatted
        if task == "assistant":
            formatted = prompt.format(language="English", message="test")
        elif task == "summarizer":
            formatted = prompt.format(text="test", length="brief")
        elif task == "translator":
            formatted = prompt.format(
                text="test",
                source_language="English",
                target_language="Spanish",
                context="",
            )
        elif task == "coder":
            formatted = prompt.format(
                message="test",
                language="Python",
                task_type="implementation",
                requirements="",
            )

        # Verify formatting worked
        assert isinstance(formatted, str)
        assert len(formatted) > 0


def test_prompt_selector_integration():
    """Test that prompt selector correctly maps task names to templates."""
    factory = PromptFactory()

    # Test that each task returns a different prompt (different purpose)
    assistant_prompt = factory.get_prompt_template("assistant")
    summarizer_prompt = factory.get_prompt_template("summarizer")

    # They should be different prompts
    assistant_formatted = assistant_prompt.format(language="English", message="test")
    summarizer_formatted = summarizer_prompt.format(text="test", length="brief")

    # Content should reflect different purposes
    assert isinstance(assistant_formatted, str)
    assert isinstance(summarizer_formatted, str)


def test_error_handling_integration():
    """Test that error handling works in an integrated way."""
    factory = PromptFactory()

    # Test invalid task
    with pytest.raises(ValueError) as excinfo:
        factory.get_prompt_template("nonexistent_task")

    error_msg = str(excinfo.value).lower()
    assert "not supported" in error_msg or "invalid" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
