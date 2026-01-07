#!/usr/bin/env python3
"""
Tests for chains module.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from chains import AdvancedChainBuilder, TranslationChain, CodeReviewChain

# Ensure the src directory is in sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAdvancedChainBuilder:
    """Test AdvancedChainBuilder class."""

    def setup_method(self):
        """Setup before each test method."""
        # Mock the dependencies
        with patch('chains.BedrockClient'):
            with patch('chains.PromptFactory'):
                with patch('chains.AssistantChain'):
                    with patch('chains.ToolRegistry'):
                        with patch('chains.ConversationBuffer'):
                            self.builder = AdvancedChainBuilder()

    def test_initialization(self):
        """Test that AdvancedChainBuilder initializes correctly."""
        assert self.builder is not None
        assert hasattr(self.builder, 'client')
        assert hasattr(self.builder, 'prompt_factory')
        assert hasattr(self.builder, 'basic_chain_builder')
        assert hasattr(self.builder, 'tool_registry')
        assert hasattr(self.builder, 'memory_store')

    def test_list_available_chains(self):
        """Test that list_available_chains returns expected chain types."""
        chains = self.builder.list_available_chains()

        expected_chains = [
            "conversational",
            "tool_calling",
            "sequential",
            "conditional",
            "summarization_pipeline",
            "translation_chain",
            "code_review_chain",
        ]

        for chain_type in expected_chains:
            assert chain_type in chains
            assert isinstance(chains[chain_type], str)

    @patch('chains.RunnableSequence')
    @patch('chains.RunnablePassthrough')
    def test_create_conversational_chain(self, mock_passthrough, mock_sequence):
        """Test creating a conversational chain."""
        # Mock the basic chain
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value="Test response")
        self.builder.basic_chain_builder.create_chain = Mock(return_value=mock_chain)

        # Mock memory store
        self.builder.memory_store.get_history = Mock(return_value=[])
        self.builder.memory_store.add_message = Mock()

        chain = self.builder.create_conversational_chain(
            task="assistant",
            memory_enabled=True
        )

        assert chain is not None
        self.builder.basic_chain_builder.create_chain.assert_called_once()

    def test_get_chain_builder_info(self):
        """Test getting chain builder information."""

        # Mock self.tool_registry.list_tools() to return a dictionary
        with patch.object(self.builder.tool_registry, 'list_tools') as mock_list_tools:
            mock_list_tools.return_value = {
                "translation_chain": {
                    "description": "Translates text between languages",
                    "input": "text",
                    "output": "translated_text"
                },
                "code_review_chain": {
                    "description": "Analyzes code for improvements",
                    "input": "code_snippet",
                    "output": "review_comments"
                }
            }

            # Also mock memory_store.get_conversation_count() if needed
            with patch.object(self.builder.memory_store, 'get_conversation_count', return_value=10):

                # Now call the method
                info = self.builder.get_chain_builder_info()

                # Debug: print what we got
                print(f"DEBUG: tools_registered type: {type(info.get('tools_registered'))}")
                print(f"DEBUG: tools_registered value: {info.get('tools_registered')}")

                # Your assertions
                assert "available_chains" in info
                assert "tools_registered" in info
                assert "memory_enabled" in info
                assert "max_conversations" in info

                assert isinstance(info["available_chains"], dict)
                assert isinstance(info["tools_registered"], dict)
                assert isinstance(info["memory_enabled"], bool)
                assert isinstance(info["max_conversations"], int)


class TestTranslationChain:
    """Test TranslationChain class."""

    def setup_method(self):
        """Setup before each test method."""
        with patch('chains.BedrockClient'):
            with patch('chains.PromptFactory'):
                self.translation_chain = TranslationChain()

    def test_initialization(self):
        """Test that TranslationChain initializes correctly."""
        assert self.translation_chain is not None
        assert hasattr(self.translation_chain, 'client')
        assert hasattr(self.translation_chain, 'prompt_factory')

    @patch('chains.AssistantChain')
    @patch('chains.RunnableSequence')
    @patch('chains.RunnablePassthrough')
    def test_create_multi_step_translation_chain(self, mock_passthrough, mock_sequence, mock_assistant_chain):
        """Test creating a multi-step translation chain."""
        chain = self.translation_chain.create_multi_step_translation_chain(
            preserve_formatting=True,
            verify_translation=False
        )

        assert chain is not None


class TestCodeReviewChain:
    """Test CodeReviewChain class."""

    def setup_method(self):
        """Setup before each test method."""
        with patch('chains.BedrockClient'):
            with patch('chains.PromptFactory'):
                self.code_review_chain = CodeReviewChain()

    def test_initialization(self):
        """Test that CodeReviewChain initializes correctly."""
        assert self.code_review_chain is not None
        assert hasattr(self.code_review_chain, 'client')
        assert hasattr(self.code_review_chain, 'prompt_factory')

    @patch('chains.ChatPromptTemplate')
    @patch('chains.StrOutputParser')
    def test_create_code_review_chain(self, mock_output_parser, mock_prompt_template):
        """Test creating a code review chain."""
        mock_model = Mock()
        self.code_review_chain.client.create_chat_model = Mock(return_value=mock_model)

        chain = self.code_review_chain.create_code_review_chain(
            language="Python",
            review_aspects=["syntax", "style"]
        )

        assert chain is not None
        self.code_review_chain.client.create_chat_model.assert_called_once()


def test_chains_module_imports():
    """Test that chains module imports correctly."""
    from chains import AdvancedChainBuilder, TranslationChain, CodeReviewChain

    assert AdvancedChainBuilder is not None
    assert TranslationChain is not None
    assert CodeReviewChain is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
