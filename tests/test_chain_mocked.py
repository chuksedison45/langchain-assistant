#!/usr/bin/env python3
"""
Mock tests for chain functionality.
These tests mock AWS Bedrock to test chain creation without real API calls.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock AWS/Bedrock before imports
with patch('boto3.Session'), \
     patch('boto3.client'), \
     patch('langchain_aws.ChatBedrock'):
    
    from chain import AssistantChain
    from prompts import PromptFactory


class TestAssistantChainMocked:
    """Test AssistantChain with mocked AWS dependencies."""
    
    def setup_method(self):
        """Setup before each test method."""
        # Create a completely mocked chain builder
        with patch('chain.BedrockClient') as mock_client_class:
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            
            # Mock the create_chat_model method
            self.mock_model = Mock()
            self.mock_client.create_chat_model.return_value = self.mock_model
            
            self.chain_builder = AssistantChain()
    
    def test_chain_initialization(self):
        """Test that chain initializes with mocked client."""
        assert self.chain_builder is not None
        assert hasattr(self.chain_builder, 'client')
        assert hasattr(self.chain_builder, 'prompt_factory')
        assert hasattr(self.chain_builder, 'chain_cache')
    
    def test_create_chain_with_valid_task(self):
        """Test create_chain with valid task name."""
        chain = self.chain_builder.create_chain(task="assistant")
        
        # Chain should be created (though mocked)
        assert chain is not None
        
        # Verify client was called to create model
        self.mock_client.create_chat_model.assert_called_once()
    
    def test_create_chain_with_invalid_task(self):
        """Test create_chain with invalid task raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            self.chain_builder.create_chain(task="invalid_task")
        
        assert "not supported" in str(excinfo.value).lower()
    
    @patch('chain.ChatPromptTemplate')
    @patch('chain.StrOutputParser')
    def test_chain_components(self, mock_output_parser, mock_prompt_template):
        """Test that chain is built with correct components."""
        # Mock the prompt factory
        mock_prompt = Mock()
        mock_prompt_factory = Mock()
        mock_prompt_factory.get_prompt_template.return_value = mock_prompt
        mock_prompt_factory.SUPPORTED_TASKS = {"assistant": "Test assistant"}
        
        # Replace the prompt factory
        self.chain_builder.prompt_factory = mock_prompt_factory
        
        # Create chain
        chain = self.chain_builder.create_chain(task="assistant")
        
        # Verify prompt factory was called
        mock_prompt_factory.get_prompt_template.assert_called_once_with("assistant")
        
        # Model creation should have been called
        self.mock_client.create_chat_model.assert_called_once()
    
    def test_get_chain_caching(self):
        """Test that get_chain caches chains."""
        # Create first chain
        chain1 = self.chain_builder.get_chain(task="assistant")
        
        # Reset mock to track new calls
        self.mock_client.create_chat_model.reset_mock()
        
        # Get chain again - should use cache
        chain2 = self.chain_builder.get_chain(task="assistant")
        
        # Should not create new model (should use cache)
        self.mock_client.create_chat_model.assert_not_called()
        
        # Both should be the same object (from cache)
        assert chain1 is chain2
    
    def test_get_chain_different_tasks(self):
        """Test that different tasks create different chains."""
        # Create chain for assistant
        assistant_chain = self.chain_builder.get_chain(task="assistant")
        
        # Reset mock
        self.mock_client.create_chat_model.reset_mock()
        
        # Create chain for summarizer
        summarizer_chain = self.chain_builder.get_chain(task="summarizer")
        
        # Should create new model for different task
        self.mock_client.create_chat_model.assert_called_once()
        
        # Chains should be different
        assert assistant_chain is not summarizer_chain
    
    def test_create_chat_chain(self):
        """Test create_chat_chain convenience method."""
        chain = self.chain_builder.create_chat_chain()
        
        assert chain is not None
        self.mock_client.create_chat_model.assert_called_once()
    
    def test_create_summarizer_chain(self):
        """Test create_summarizer_chain with different lengths."""
        for length in ["brief", "medium", "detailed"]:
            # Reset mock for each iteration
            self.mock_client.create_chat_model.reset_mock()
            
            chain = self.chain_builder.create_summarizer_chain(length=length)
            
            assert chain is not None
            self.mock_client.create_chat_model.assert_called_once()


class TestPromptSelector:
    """Test prompt selector functionality."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.factory = PromptFactory()
    
    def test_prompt_selector_returns_correct_type(self):
        """Test that prompt selector returns correct template type for each task."""
        for task in self.factory.SUPPORTED_TASKS:
            prompt = self.factory.get_prompt_template(task)
            assert isinstance(prompt, ChatPromptTemplate)
    
    def test_prompt_selector_with_kwargs(self):
        """Test prompt selector passes kwargs to prompt creators."""
        # Test with summarizer length parameter
        prompt = self.factory.get_prompt_template("summarizer", length="brief")
        
        # Format to verify length parameter was used
        formatted = prompt.format(text="test", length="brief")
        assert "brief" in formatted.lower()
        
        # Test with assistant include_examples parameter
        prompt_with_examples = self.factory.get_prompt_template(
            "assistant", 
            include_examples=True
        )
        formatted_with_examples = prompt_with_examples.format(
            language="English", 
            message="test"
        )
        assert "examples" in formatted_with_examples.lower()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])