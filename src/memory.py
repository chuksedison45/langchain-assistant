"""
Memory module for managing conversation history and context.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pickle
import os
from collections import defaultdict


class ConversationBuffer:
    """
    Simple conversation buffer for storing conversation history.
    """

    def __init__(self, max_conversations: int = 100, max_messages_per_conversation: int = 100):
        """
        Initialize conversation buffer.

        Args:
            max_conversations: Maximum number of conversations to store
            max_messages_per_conversation: Maximum messages per conversation
        """
        self.max_conversations = max_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}

    def add_message(
        self,
        conversation_id: str,
        message: Dict[str, Any],
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a message to the conversation buffer.

        Args:
            conversation_id: Unique identifier for the conversation
            message: Message content (dict with 'role' and 'content')
            role: Override role if not in message
            timestamp: Optional timestamp (defaults to now)
        """
        if len(self.conversations) >= self.max_conversations:
            # Remove oldest conversation if limit reached
            self._remove_oldest_conversation()

        if conversation_id not in self.conversation_metadata:
            self.conversation_metadata[conversation_id] = {
                "created": datetime.now(),
                "updated": datetime.now(),
                "message_count": 0,
            }

        # Ensure message has required fields
        if isinstance(message, dict):
            if role is not None:
                message["role"] = role
            if "role" not in message:
                message["role"] = "user"
            if "content" not in message:
                message["content"] = str(message)
            if "timestamp" not in message:
                message["timestamp"] = timestamp or datetime.now()
        else:
            message = {
                "role": role or "user",
                "content": str(message),
                "timestamp": timestamp or datetime.now(),
            }

        # Add to conversation
        conversation = self.conversations[conversation_id]
        if len(conversation) >= self.max_messages_per_conversation:
            conversation.pop(0)  # Remove oldest message

        conversation.append(message)
        self.conversation_metadata[conversation_id]["updated"] = datetime.now()
        self.conversation_metadata[conversation_id]["message_count"] += 1

    def get_history(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        recent_first: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum number of messages to return
            recent_first: Whether to return most recent messages first

        Returns:
            List of message dictionaries
        """
        if conversation_id not in self.conversations:
            return []

        conversation = self.conversations[conversation_id]

        if recent_first:
            conversation = list(reversed(conversation))

        if max_messages is not None:
            conversation = conversation[:max_messages]

        return conversation

    def get_formatted_history(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        format_str: str = "{role}: {content}"
    ) -> str:
        """
        Get formatted conversation history as string.

        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum messages to include
            format_str: Format string for each message

        Returns:
            Formatted conversation history
        """
        history = self.get_history(conversation_id, max_messages)
        formatted = []

        for message in history:
            formatted.append(format_str.format(**message))

        return "\n".join(formatted)

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear all messages from a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.conversation_metadata:
            del self.conversation_metadata[conversation_id]

    def clear_all(self) -> None:
        """Clear all conversations."""
        self.conversations.clear()
        self.conversation_metadata.clear()

    def get_conversation_count(self) -> int:
        """Get number of active conversations."""
        return len(self.conversations)

    def get_total_messages(self) -> int:
        """Get total number of messages across all conversations."""
        return sum(len(conv) for conv in self.conversations.values())

    def _remove_oldest_conversation(self) -> None:
        """Remove the conversation with oldest update time."""
        if not self.conversation_metadata:
            return

        oldest_id = min(
            self.conversation_metadata.items(),
            key=lambda x: x[1]["updated"]
        )[0]

        self.clear_conversation(oldest_id)

    def save_to_file(self, filepath: str, format: str = "json") -> None:
        """
        Save conversations to file.

        Args:
            filepath: Path to save file
            format: File format ('json' or 'pickle')
        """
        data = {
            "conversations": dict(self.conversations),
            "metadata": self.conversation_metadata,
            "config": {
                "max_conversations": self.max_conversations,
                "max_messages_per_conversation": self.max_messages_per_conversation,
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format == "json":
            # Convert datetime objects to strings for JSON
            def datetime_converter(o):
                if isinstance(o, datetime):
                    return o.isoformat()
                raise TypeError(f"Object of type {type(o)} is not JSON serializable")

            with open(filepath, 'w') as f:
                json.dump(data, f, default=datetime_converter, indent=2)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_from_file(self, filepath: str, format: str = "json") -> None:
        """
        Load conversations from file.

        Args:
            filepath: Path to load file
            format: File format ('json' or 'pickle')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format == "pickle":
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Convert string timestamps back to datetime objects
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        try:
                            obj[key] = datetime.fromisoformat(value)
                        except (ValueError, AttributeError):
                            pass
                    elif isinstance(value, (list, dict)):
                        convert_timestamps(value)
            elif isinstance(obj, list):
                for item in obj:
                    convert_timestamps(item)

        convert_timestamps(data)

        self.conversations = defaultdict(list, data.get("conversations", {}))
        self.conversation_metadata = data.get("metadata", {})

        config = data.get("config", {})
        self.max_conversations = config.get("max_conversations", 100)
        self.max_messages_per_conversation = config.get("max_messages_per_conversation", 100)


class SummaryMemory:
    """
    Memory that stores summarized conversation history.
    Useful for long conversations where full history is too large.
    """

    def __init__(self, summary_interval: int = 10):
        """
        Initialize summary memory.

        Args:
            summary_interval: Number of messages after which to summarize
        """
        self.summary_interval = summary_interval
        self.conversation_summaries: Dict[str, List[str]] = defaultdict(list)
        self.recent_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Add message and summarize if needed.

        Args:
            conversation_id: Conversation identifier
            message: Message dictionary
        """
        self.recent_messages[conversation_id].append(message)

        # Summarize if we have enough messages
        if len(self.recent_messages[conversation_id]) >= self.summary_interval:
            self._summarize_conversation(conversation_id)

    def _summarize_conversation(self, conversation_id: str) -> None:
        """Summarize recent messages in a conversation."""
        messages = self.recent_messages[conversation_id]
        if not messages:
            return

        # Create summary (in real implementation, this would use an LLM)
        summary = f"Summary of {len(messages)} messages: " + \
                  " ".join(str(msg.get("content", ""))[:50] for msg in messages)

        self.conversation_summaries[conversation_id].append(summary)
        self.recent_messages[conversation_id].clear()

    def get_context(self, conversation_id: str) -> str:
        """
        Get context for conversation (summaries + recent messages).

        Args:
            conversation_id: Conversation identifier

        Returns:
            Context string
        """
        context_parts = []

        # Add summaries
        if conversation_id in self.conversation_summaries:
            for i, summary in enumerate(self.conversation_summaries[conversation_id], 1):
                context_parts.append(f"Summary {i}: {summary}")

        # Add recent messages
        if conversation_id in self.recent_messages:
            for msg in self.recent_messages[conversation_id]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)
