#!/usr/bin/env python3
"""
Mock tests for chain functionality.
These tests mock AWS Bedrock to test chain creation without real API calls.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from langchain_core.prompts import ChatPromptTemplate

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestAssistantChainMocked:
    """Test AssistantChain with mocked AWS dependencies."""

    def setup_method(self):
        """Setup before each test method."""
        # Mock the BedrockClient before importing chain
        with patch("chain.BedrockClient") as mock_client_class:
            # Also mock the ChatBedrock import
            with patch("langchain_aws.ChatBedrock"):
                # Now import the chain module
                from chain import AssistantChain

                self.mock_client = Mock()
                mock_client_class.return_value = self.mock_client

                # Mock the create_chat_model method
                self.mock_model = Mock()
                self.mock_client.create_chat_model.return_value = self.mock_model

                # Mock the prompt factory
                with patch("chain.PromptFactory") as mock_factory_class:
                    self.mock_factory = Mock()
                    mock_factory_class.return_value = self.mock_factory
                    self.mock_factory.SUPPORTED_TASKS = {
                        "assistant": "Test assistant",
                        "summarizer": "Test summarizer",
                    }

                    # Create the chain builder
                    self.chain_builder = AssistantChain()

    def test_chain_initialization(self):
        """Test that chain initializes with mocked client."""
        assert self.chain_builder is not None
        assert hasattr(self.chain_builder, "client")
        assert hasattr(self.chain_builder, "prompt_factory")
        assert hasattr(self.chain_builder, "chain_cache")

    def test_create_chain_with_valid_task(self):
        """Test create_chain with valid task name."""
        # Mock the prompt factory method
        mock_prompt = Mock()
        self.mock_factory.get_prompt_template.return_value = mock_prompt

        # Mock the | operator behavior
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_chain.__or__ = Mock(return_value=mock_chain)

        chain = self.chain_builder.create_chain(task="assistant")

        # Chain should be created (though mocked)
        assert chain is not None

        # Verify client was called to create model
        self.mock_client.create_chat_model.assert_called_once()
        self.mock_factory.get_prompt_template.assert_called_once()

    def test_create_chain_with_invalid_task(self):
        """Test create_chain with invalid task raises ValueError."""
        # Mock the prompt factory to raise ValueError
        self.mock_factory.get_prompt_template.side_effect = ValueError(
            """Task 'invalid_task' not supported. Available tasks:
              ['assistant', 'summarizer']"""
        )

        with pytest.raises(ValueError) as excinfo:
            self.chain_builder.create_chain(task="invalid_task")

        # Check error message contains expected text
        error_msg = str(excinfo.value)
        # The actual error message is: "Task 'invalid_task' not supported.
        # Available tasks: ['assistant', 'summarizer']"
        # So we check for "not supported" instead of "Invalid"
        assert "not supported" in error_msg.lower()

    @patch("chain.StrOutputParser")
    def test_chain_components(self, mock_output_parser):
        """Test that chain is built with correct components."""
        # Mock the prompt
        mock_prompt = Mock()
        self.mock_factory.get_prompt_template.return_value = mock_prompt

        # Mock the output parser
        mock_parser_instance = Mock()
        mock_output_parser.return_value = mock_parser_instance

        # Mock the | operator
        mock_chain_part1 = Mock()
        mock_chain_part2 = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain_part1)
        mock_chain_part1.__or__ = Mock(return_value=mock_chain_part2)

        # Create chain
        chain = self.chain_builder.create_chain(task="assistant")
        # Add assertions about the chain
        assert chain is not None
        assert chain == mock_chain_part2  # Verify the chain matches our mock

        # Verify prompt factory was called
        self.mock_factory.get_prompt_template.assert_called_once_with("assistant")

        # Model creation should have been called
        self.mock_client.create_chat_model.assert_called_once()

        # Output parser should have been used
        mock_output_parser.assert_called_once()

    def test_get_chain_caching(self):
        """Test that get_chain caches chains."""
        # Mock the prompt
        mock_prompt = Mock()
        self.mock_factory.get_prompt_template.return_value = mock_prompt

        # Mock the | operator
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_chain.__or__ = Mock(return_value=mock_chain)

        # Create first chain
        chain1 = self.chain_builder.get_chain(task="assistant")

        # Reset mocks to track new calls
        self.mock_client.create_chat_model.reset_mock()
        self.mock_factory.get_prompt_template.reset_mock()

        # Get chain again - should use cache
        chain2 = self.chain_builder.get_chain(task="assistant")
        # Verify it's the SAME object from cache
        assert chain1 is chain2, "Should return cached chain object"

        # Should not create new model (should use cache)
        self.mock_client.create_chat_model.assert_not_called()

        # Should not call get_prompt_template again (cached)
        self.mock_factory.get_prompt_template.assert_not_called()

    def test_create_chat_chain(self):
        """Test create_chat_chain convenience method."""
        # Mock the prompt
        mock_prompt = Mock()
        self.mock_factory.get_prompt_template.return_value = mock_prompt

        # Mock the | operator
        mock_chain = Mock()
        mock_prompt.__or__ = Mock(return_value=mock_chain)
        mock_chain.__or__ = Mock(return_value=mock_chain)

        chain = self.chain_builder.create_chat_chain()

        assert chain is not None
        self.mock_client.create_chat_model.assert_called_once()
        self.mock_factory.get_prompt_template.assert_called_once()


class TestPromptSelector:
    """Test prompt selector functionality."""

    def setup_method(self):
        """Setup before each test method."""
        # Import the actual PromptFactory for these tests
        from prompts import PromptFactory

        self.factory = PromptFactory()

    def test_prompt_selector_returns_correct_type(self):
        """Test that prompt selector returns correct template type for each task."""
        for task in self.factory.SUPPORTED_TASKS:
            prompt = self.factory.get_prompt_template(task)
            assert isinstance(prompt, ChatPromptTemplate)

    def test_prompt_selector_with_kwargs(self):
        """Test prompt selector passes kwargs to prompt creators."""
        # Test with summarizer - it should accept length parameter in format, 
        # not in get_prompt_template
        prompt = self.factory.get_prompt_template("summarizer")

        # Format to verify it works with length parameter
        formatted = prompt.format(text="test", length="brief")
        assert isinstance(formatted, str)
        assert len(formatted) > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
