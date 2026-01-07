#!/usr/bin/env python3
"""
Tests for memory module.
"""
import pytest
import sys
import os
import tempfile
from memory import ConversationBuffer, SummaryMemory
# Ensure the src directory is in sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConversationBuffer:
    """Test ConversationBuffer class."""

    def setup_method(self):
        """Setup before each test method."""
        self.buffer = ConversationBuffer(max_conversations=5, max_messages_per_conversation=3)

    def test_initialization(self):
        """Test that ConversationBuffer initializes correctly."""
        assert self.buffer is not None
        assert self.buffer.max_conversations == 5
        assert self.buffer.max_messages_per_conversation == 3
        assert len(self.buffer.conversations) == 0

    def test_add_message_dict(self):
        """Test adding a message as dictionary."""
        conversation_id = "test_conv_1"
        message = {"role": "user", "content": "Hello, world!"}

        self.buffer.add_message(conversation_id, message)

        assert conversation_id in self.buffer.conversations
        assert len(self.buffer.conversations[conversation_id]) == 1

        stored_message = self.buffer.conversations[conversation_id][0]
        assert stored_message["role"] == "user"
        assert stored_message["content"] == "Hello, world!"
        assert "timestamp" in stored_message

    def test_add_message_string(self):
        """Test adding a message as string."""
        conversation_id = "test_conv_2"
        message = "Simple string message"

        self.buffer.add_message(conversation_id, message, role="assistant")

        stored_message = self.buffer.conversations[conversation_id][0]
        assert stored_message["role"] == "assistant"
        assert stored_message["content"] == "Simple string message"

    def test_get_history(self):
        """Test getting conversation history."""
        conversation_id = "test_conv_3"

        # Add multiple messages
        for i in range(3):
            self.buffer.add_message(conversation_id, {"role": "user", "content": f"Message {i}"})

        history = self.buffer.get_history(conversation_id)

        assert len(history) == 3
        assert history[0]["content"] == "Message 0"
        assert history[2]["content"] == "Message 2"

    def test_get_formatted_history(self):
        """Test getting formatted conversation history."""
        conversation_id = "test_conv_4"

        self.buffer.add_message(conversation_id, {"role": "user", "content": "Hello"})
        self.buffer.add_message(conversation_id, {"role": "assistant", "content": "Hi there!"})

        formatted = self.buffer.get_formatted_history(conversation_id)

        assert "user: Hello" in formatted
        assert "assistant: Hi there!" in formatted

    def test_clear_conversation(self):
        """Test clearing a conversation."""
        conversation_id = "test_conv_5"

        self.buffer.add_message(conversation_id, {"role": "user", "content": "Test"})
        assert conversation_id in self.buffer.conversations

        self.buffer.clear_conversation(conversation_id)
        assert conversation_id not in self.buffer.conversations

    def test_clear_all(self):
        """Test clearing all conversations."""
        for i in range(3):
            self.buffer.add_message(f"conv_{i}", {"role": "user", "content": f"Message {i}"})

        assert self.buffer.get_conversation_count() == 3

        self.buffer.clear_all()
        assert self.buffer.get_conversation_count() == 0

    def test_max_messages_limit(self):
        """Test that max messages per conversation is enforced."""
        conversation_id = "test_conv_6"

        # Add more messages than limit
        for i in range(5):
            self.buffer.add_message(conversation_id, {"role": "user", "content": f"Message {i}"})

        # Should only keep the last 3 messages
        history = self.buffer.get_history(conversation_id)
        assert len(history) == 3
        assert history[0]["content"] == "Message 2"  # Oldest kept message
        assert history[2]["content"] == "Message 4"  # Newest message

    def test_save_and_load_json(self):
        """Test saving and loading from JSON file."""
        conversation_id = "test_conv_7"

        # Add some messages
        self.buffer.add_message(conversation_id, {"role": "user", "content": "Save test"})

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            self.buffer.save_to_file(temp_file, format="json")

            # Create new buffer and load
            new_buffer = ConversationBuffer()
            new_buffer.load_from_file(temp_file, format="json")

            assert new_buffer.get_conversation_count() == 1
            history = new_buffer.get_history(conversation_id)
            assert len(history) == 1
            assert history[0]["content"] == "Save test"

        finally:
            # Clean up
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_get_total_messages(self):
        """Test getting total messages count."""
        # Add messages to multiple conversations
        self.buffer.add_message("conv_a", {"role": "user", "content": "Message 1"})
        self.buffer.add_message("conv_a", {"role": "assistant", "content": "Message 2"})
        self.buffer.add_message("conv_b", {"role": "user", "content": "Message 3"})

        total = self.buffer.get_total_messages()
        assert total == 3


class TestSummaryMemory:
    """Test SummaryMemory class."""

    def setup_method(self):
        """Setup before each test method."""
        self.memory = SummaryMemory(summary_interval=3)

    def test_initialization(self):
        """Test that SummaryMemory initializes correctly."""
        assert self.memory is not None
        assert self.memory.summary_interval == 3
        assert len(self.memory.conversation_summaries) == 0
        assert len(self.memory.recent_messages) == 0

    def test_add_message(self):
        """Test adding messages to memory."""
        conversation_id = "test_conv"

        # Add messages (less than summary interval)
        for i in range(2):
            self.memory.add_message(conversation_id, {"role": "user", "content": f"Message {i}"})

        # Should not have summarized yet
        assert len(self.memory.recent_messages[conversation_id]) == 2
        assert len(self.memory.conversation_summaries.get(conversation_id, [])) == 0

        # Add one more message to trigger summarization
        self.memory.add_message(conversation_id, {"role": "assistant", "content": "Message 2"})

        # Should have summarized and cleared recent messages
        assert len(self.memory.recent_messages[conversation_id]) == 0
        assert len(self.memory.conversation_summaries[conversation_id]) == 1

    def test_get_context(self):
        """Test getting context from memory."""
        conversation_id = "test_conv_2"

        # Add some messages
        for i in range(4):
            self.memory.add_message(conversation_id, {"role": "user", "content": f"Message {i}"})

        context = self.memory.get_context(conversation_id)

        # Should include summary and recent messages
        assert "Summary" in context
        assert "user: Message" in context


def test_memory_module_imports():
    """Test that memory module imports correctly."""
    from memory import ConversationBuffer, SummaryMemory

    assert ConversationBuffer is not None
    assert SummaryMemory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
