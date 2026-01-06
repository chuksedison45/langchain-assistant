from typing import Optional, Dict, Literal
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from src.bedrock_client import BedrockClient
from src.prompts import PromptFactory


class AssistantChain:
    """Build and manage LangChain chains for various tasks with output parsing."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """Initialize with optional custom Bedrock client."""
        self.client = bedrock_client or BedrockClient()
        self.prompt_factory = PromptFactory()
        self.chain_cache: Dict[str, RunnableSequence] = {}

    def create_chain(
        self,
        task: str = "assistant",
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **prompt_kwargs,
    ) -> RunnableSequence:
        """
        Create a LangChain chain for a specific task using LCEL with output parsing.

        Args:
            task: The task name (assistant, summarizer, translator, etc.)
            model_id: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **prompt_kwargs: Additional parameters for the prompt

        Returns:
            RunnableSequence chain with output parsing
        """
        # Validate task
        if task not in self.prompt_factory.SUPPORTED_TASKS:
            available = list(self.prompt_factory.SUPPORTED_TASKS.keys())
            raise ValueError(
                f"Task '{task}' not supported. Available tasks: {available}"
            )

        # Create model
        model = self.client.create_chat_model(
            model_id=model_id, temperature=temperature, max_tokens=max_tokens
        )

        # Get appropriate prompt template
        prompt = self.prompt_factory.get_prompt_template(task, **prompt_kwargs)

        # Use StrOutputParser for clean text output
        output_parser = StrOutputParser()

        # Build chain using LCEL: prompt → model → output parser
        chain = prompt | model | output_parser

        task_description = self.prompt_factory.SUPPORTED_TASKS[task]
        print(f"✓ Chain created for task: {task} ({task_description})")
        print(f"  Output parser: {output_parser.__class__.__name__}")

        return chain

    def create_chat_chain(self, **kwargs) -> RunnableSequence:
        """Create a chain for chat conversations with output parsing."""
        return self.create_chain(task="assistant", **kwargs)

    def create_summarizer_chain(
        self, length: Literal["brief", "medium", "detailed"] = "medium", **kwargs
    ) -> RunnableSequence:
        """Create a specialized summarizer chain with output parsing."""
        return self.create_chain(task="summarizer", length=length, **kwargs)

    def get_chain(
        self, task: str = "assistant", cache_key: Optional[str] = None, **kwargs
    ) -> RunnableSequence:
        """
        Get a cached chain or create a new one.

        Args:
            task: The task name
            cache_key: Optional custom cache key (defaults to task)
            **kwargs: Chain creation parameters

        Returns:
            Cached or new chain
        """
        key = cache_key or task

        if key not in self.chain_cache:
            self.chain_cache[key] = self.create_chain(task, **kwargs)

        return self.chain_cache[key]
