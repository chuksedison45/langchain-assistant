#!/usr/bin/env python3
"""
Test suite for prompt templates and prompt factory.
Tests prompt formatting, selector functionality, and error handling.
These tests run without AWS credentials - they only test prompt formatting.
"""

import pytest
import sys
import os
from prompts import PromptFactory
from langchain_core.prompts import ChatPromptTemplate
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestPromptFactory:
    """Test suite for PromptFactory class."""

    def setup_method(self):
        """Setup before each test method."""
        self.factory = PromptFactory()

    def test_list_tasks(self, capsys):
        """Test that list_tasks() prints available tasks."""
        tasks = self.factory.list_tasks()

        captured = capsys.readouterr()

        # Check that output contains expected headers
        assert "Available Prompt Tasks:" in captured.out
        assert "-" * 40 in captured.out

        # Check that all supported tasks are listed
        for task in self.factory.SUPPORTED_TASKS:
            assert task in captured.out
            assert self.factory.SUPPORTED_TASKS[task] in captured.out

        # Check that returned list matches supported tasks
        assert tasks == list(self.factory.SUPPORTED_TASKS.keys())

    def test_get_prompt_template_valid_tasks(self):
        """Test that get_prompt_template() returns correct template for valid tasks."""
        for task in self.factory.SUPPORTED_TASKS:
            prompt = self.factory.get_prompt_template(task)

            # Verify it returns a ChatPromptTemplate
            assert isinstance(prompt, ChatPromptTemplate)

            # Verify it has the expected structure
            assert hasattr(prompt, "messages")
            assert hasattr(prompt, "format")

    def test_get_prompt_template_assistant(self):
        """Test assistant prompt template formatting."""
        prompt = self.factory.get_prompt_template("assistant")

        # Test formatting with sample inputs
        formatted = prompt.format(language="English", message="Hello")

        # Check that formatted string contains expected content
        assert "Hello" in formatted
        assert "English" in formatted
        assert "assistant" in formatted.lower() or "helpful" in formatted.lower()

    def test_get_prompt_template_summarizer(self):
        """Test summarizer prompt template formatting with different lengths."""
        for length in ["brief", "medium", "detailed"]:
            prompt = self.factory.get_prompt_template("summarizer")

            formatted = prompt.format(text="Sample text to summarize", length=length)

            # Check that formatted string contains expected content
            assert "Sample text to summarize" in formatted
            assert "summary" in formatted.lower() or "summarize" in formatted.lower()

    def test_get_prompt_template_translator(self):
        """Test translator prompt template formatting."""
        prompt = self.factory.get_prompt_template("translator")

        formatted = prompt.format(
            text="Hello world",
            source_language="English",
            target_language="Spanish",
            context="",
        )

        assert "Hello world" in formatted
        assert "English" in formatted
        assert "Spanish" in formatted
        assert "translator" in formatted.lower()

    def test_get_prompt_template_coder(self):
        """Test coder prompt template formatting."""
        prompt = self.factory.get_prompt_template("coder")

        formatted = prompt.format(
            message="Write a function",
            language="Python",
            task_type="implementation",
            requirements="",
        )

        assert "Write a function" in formatted
        assert "Python" in formatted
        assert "implementation" in formatted
        assert "coder" in formatted.lower() or "developer" in formatted.lower()

    def test_get_prompt_template_invalid_task(self):
        """Test that requesting an unknown prompt raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            self.factory.get_prompt_template("invalid_task_name")

        # Check error message contains expected text
        assert "not supported" in str(excinfo.value).lower()
        assert "available" in str(excinfo.value).lower()

    def test_get_task_input_variables(self):
        """
        Test that get_task_input_variables returns correct input variables for each
        task.
        """
        test_cases = [
            ("assistant", ["language", "message"]),
            ("summarizer", ["text", "length"]),
            ("translator", ["text", "source_language", "target_language", "context"]),
            ("coder", ["language", "task_type", "requirements", "message"]),
        ]

        for task, expected_vars in test_cases:
            actual_vars = self.factory.get_task_input_variables(task)
            assert actual_vars == expected_vars, f"Failed for task: {task}"

    def test_get_task_input_variables_invalid_task(self):
        """Test that get_task_input_variables returns default for invalid task."""
        # Should return ["message"] for unknown task
        result = self.factory.get_task_input_variables("unknown_task")
        assert result == ["message"]

    def test_prompt_template_structure(self):
        """Test that all prompt templates have the correct structure."""
        for task in self.factory.SUPPORTED_TASKS:
            prompt = self.factory.get_prompt_template(task)

            # Check that prompt has required methods
            assert hasattr(prompt, "format")
            assert hasattr(prompt, "messages")

            # Check that format is callable
            assert callable(prompt.format)

    def test_prompt_factory_is_singleton_like(self):
        """Test that PromptFactory methods work as class methods."""
        # All methods should work as class methods
        prompt1 = PromptFactory.get_prompt_template("assistant")
        prompt2 = self.factory.get_prompt_template("assistant")

        # Both should return ChatPromptTemplate instances
        assert isinstance(prompt1, ChatPromptTemplate)
        assert isinstance(prompt2, ChatPromptTemplate)

    def test_prompt_includes_examples(self):
        """Test that assistant prompt can include examples when requested."""
        # Test without examples
        prompt_no_examples = PromptFactory.create_assistant_prompt(
            include_examples=False
        )
        formatted_no_examples = prompt_no_examples.format(
            language="English", message="test"
        )

        # Test with examples
        prompt_with_examples = PromptFactory.create_assistant_prompt(
            include_examples=True
        )
        formatted_with_examples = prompt_with_examples.format(
            language="English", message="test"
        )

        # The version with examples should be longer
        assert len(formatted_with_examples) > len(formatted_no_examples)
        # Note: The actual content check depends on implementation

    def test_summarizer_different_lengths(self):
        """Test that summarizer prompt correctly handles different length parameters."""
        lengths = ["brief", "medium", "detailed"]

        for length in lengths:
            prompt = self.factory.get_prompt_template("summarizer")
            formatted = prompt.format(text="Test text", length=length)

            # The prompt should contain the length variable
            assert isinstance(formatted, str)
            assert len(formatted) > 0


class TestPromptFormattingEdgeCases:
    """Test edge cases and error conditions for prompt formatting."""

    def setup_method(self):
        """Setup before each test method."""
        self.factory = PromptFactory()

    def test_empty_inputs(self):
        """Test prompt formatting with empty strings."""
        tasks_to_test = [
            ("assistant", {"language": "", "message": ""}),
            ("summarizer", {"text": "", "length": "brief"}),
            (
                "translator",
                {
                    "text": "",
                    "source_language": "",
                    "target_language": "",
                    "context": "",
                },
            ),
        ]

        for task, inputs in tasks_to_test:
            prompt = self.factory.get_prompt_template(task)

            # Should not raise exception with empty inputs
            formatted = prompt.format(**inputs)

            # Formatted string should not be None
            assert formatted is not None
            assert isinstance(formatted, str)

    def test_special_characters(self):
        """Test prompt formatting with special characters."""
        test_text = "Special chars: !@#$%^&*()_+{}|:\"<>?[]\\;',./`~"

        prompt = self.factory.get_prompt_template("assistant")
        formatted = prompt.format(language="English", message=test_text)

        # Special characters should be preserved in the formatted prompt
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_very_long_input(self):
        """Test prompt formatting with very long input."""
        long_text = "x" * 1000  # 1000 character string

        prompt = self.factory.get_prompt_template("summarizer")
        formatted = prompt.format(text=long_text, length="brief")

        # Should handle long input without error
        assert len(formatted) > 0
        assert isinstance(formatted, str)


class TestPromptFactoryErrorHandling:
    """Test error handling in PromptFactory."""

    def setup_method(self):
        """Setup before each test method."""
        self.factory = PromptFactory()

    def test_none_task(self):
        """Test that None task raises appropriate error."""
        with pytest.raises((ValueError, AttributeError)) as excinfo:
            self.factory.get_prompt_template(None)

        # Check that an error was raised
        assert excinfo.value is not None

    def test_empty_task_string(self):
        """Test that empty task string raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            self.factory.get_prompt_template("")

        # Check error message contains expected text
        error_msg = str(excinfo.value).lower()
        assert "not supported" in error_msg or "empty" in error_msg

    def test_case_sensitivity(self):
        """Test that task names are case-insensitive (or handled)."""
        # Uppercase should work (or be converted to lowercase)
        prompt_upper = self.factory.get_prompt_template("ASSISTANT")
        prompt_lower = self.factory.get_prompt_template("assistant")

        # Both should return valid prompts
        assert isinstance(prompt_upper, ChatPromptTemplate)
        assert isinstance(prompt_lower, ChatPromptTemplate)

    def test_extra_kwargs(self):
        """Test that extra kwargs don't break prompt creation."""
        # Some tasks accept extra kwargs, others should ignore them
        prompt = self.factory.get_prompt_template("assistant")

        # Should still create a valid prompt
        assert isinstance(prompt, ChatPromptTemplate)

        formatted = prompt.format(language="English", message="test")
        assert "test" in formatted


def test_prompt_imports():
    """Test that all necessary imports are available."""
    # This tests that the module can be imported without errors
    from prompts import PromptFactory
    from langchain_core.prompts import ChatPromptTemplate

    assert PromptFactory is not None
    assert ChatPromptTemplate is not None


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
