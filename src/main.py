from typing import Optional, Dict, Any, Literal
import time
from datetime import datetime
from src.chain import AssistantChain
from src.config import config
from src.bedrock_client import BedrockClient
from src.prompts import PromptFactory
from src.chains import AdvancedChainBuilder, TranslationChain, CodeReviewChain
from src.memory import ConversationBuffer
from src.tools import ToolRegistry


class LangChainAssistant:
    """Main application class for the LangChain assistant with output parsing."""

    def __init__(self, verbose: bool = True):
        """Initialize the assistant."""
        self.verbose = verbose
        self.client = BedrockClient()
        self.chain_builder = AssistantChain(self.client)
        self.prompt_factory = PromptFactory()
        self.advanced_builder = AdvancedChainBuilder(self.client)
        self.translation_chain = TranslationChain(self.client)
        self.code_review_chain = CodeReviewChain(self.client)
        self.tool_registry = ToolRegistry()
        self.memory = ConversationBuffer()

        # Current chain and task
        self.current_task = "assistant"
        self.current_chain = None

        # Task-specific configurations
        self.task_configs = {
            "assistant": {"language": "English"},
            "summarizer": {"length": "medium"},
            "translator": {"source_language": "auto", "target_language": "English"},
            "coder": {"language": "Python", "task_type": "implementation"},
        }

        self.interaction_history = []

        if verbose:
            config.print_config()
            print("\n" + "=" * 60)
            print("LangChain Assistant with Output Parsing")
            print("=" * 60)
            self.prompt_factory.list_tasks()
            print("\nOutput parsing is enabled with StrOutputParser")

    def set_task(self, task: str, **task_kwargs) -> "LangChainAssistant":
        """Set the current task and configure the appropriate chain."""

        if task not in self.prompt_factory.SUPPORTED_TASKS:
            available = list(self.prompt_factory.SUPPORTED_TASKS.keys())
            print(f"Task '{task}' not supported. Available tasks: {available}")
            return self

        self.current_task = task

        # Update task configuration
        if task in self.task_configs:
            self.task_configs[task].update(task_kwargs)
        else:
            self.task_configs[task] = task_kwargs

        # Get required input variables
        input_vars = self.prompt_factory.get_task_input_variables(task)

        # Create new chain for this task
        self.current_chain = self.chain_builder.get_chain(
            task=task, **self.task_configs[task]
        )

        if self.verbose:
            task_desc = self.prompt_factory.SUPPORTED_TASKS[task]
            print(f"\n✓ Task set to: {task} ({task_desc})")
            print(f"  Required inputs: {input_vars}")

        return self

    def process(
        self, input_data: Dict[str, Any], task: Optional[str] = None, **kwargs
    ) -> str:
        """
        Process input using the specified or current task.

        Args:
            input_data: Dictionary of input variables for the prompt
            task: Optional task name (uses current task if not specified)
            **kwargs: Additional chain parameters

        Returns:
            Clean string response from the AI
        """
        # Use specified task or current task
        target_task = task or self.current_task

        # Get or create chain
        if task and task != self.current_task:
            chain = self.chain_builder.get_chain(task=task, **kwargs)
        else:
            chain = self.current_chain or self.chain_builder.get_chain(
                task=target_task, **{**self.task_configs.get(target_task, {}), **kwargs}
            )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {target_task}")
            print(f"Input: {input_data}")
            print(f"{'='*60}\n")

        try:
            # Measure response time
            start_time = time.time()

            # Invoke the chain (returns clean string due to StrOutputParser)
            response = chain.invoke(input_data)

            # Calculate response time
            response_time = time.time() - start_time

            # Log interaction
            self.interaction_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task": target_task,
                    "input": input_data,
                    "response": response,
                    "response_time": response_time,
                }
            )

            if self.verbose:
                print(f"\n{'='*60}")
                print("Response:")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}")
                print(f"Response time: {response_time:.2f} seconds")
                print(f"Response type: {type(response).__name__}")
                print(f"{'='*60}")

            return response

        except Exception as e:
            error_msg = f"Error getting response: {e}"
            print(f"\n✗ {error_msg}")
            return error_msg

    def chat(self, message: str, language: str = "English", **kwargs) -> str:
        """Chat using assistant task with output parsing."""
        return self.process(
            {"language": language, "message": message}, task="assistant", **kwargs
        )

    def summarize(
        self,
        text: str,
        length: Literal["brief", "medium", "detailed"] = "medium",
        **kwargs,
    ) -> str:
        """Summarize text with specified length and output parsing."""
        return self.process(
            {"text": text, "length": length}, task="summarizer", **kwargs
        )

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "English",
        context: str = "",
        **kwargs,
    ) -> str:
        """Translate text between languages with output parsing."""
        return self.process(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            },
            task="translator",
            **kwargs,
        )

    def code(
        self,
        request: str,
        language: str = "Python",
        task_type: str = "implementation",
        requirements: str = "",
        **kwargs,
    ) -> str:
        """Generate or explain code with output parsing."""
        return self.process(
            {
                "message": request,
                "language": language,
                "task_type": task_type,
                "requirements": requirements,
            },
            task="coder",
            **kwargs,
        )

    def get_interaction_history(self):
        """Return the interaction history."""
        return self.interaction_history

    def clear_history(self):
        """Clear the interaction history."""
        self.interaction_history = []
        print("✓ Interaction history cleared")


def main():
    """Main entry point for the application."""

    print("=" * 60)
    print("LangChain Assistant with Output Parsing")
    print("=" * 60)

    # Create assistant
    assistant = LangChainAssistant(verbose=True)

    # Test basic chat with output parsing
    print("\n" + "=" * 60)
    print("Testing Basic Chat with Output Parsing")
    print("=" * 60)

    response = assistant.chat("What is the capital of France?", "English")

    print(f"\nResponse (clean string): {response[:100]}...")

    # Test summarizer with output parsing
    print("\n" + "=" * 60)
    print("Testing Summarizer with Output Parsing")
    print("=" * 60)

    sample_text = """
    Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly programmed.
    It focuses on the development of computer programs that can access data and use
    it to learn for themselves.
    """

    summary = assistant.summarize(sample_text, length="brief")

    print(f"\nSummary (clean string):\n{summary}")

    # Show response information
    print("\n" + "=" * 60)
    print("Response Information")
    print("=" * 60)

    print(f"Chat response type: {type(response).__name__}")
    print(f"Is string: {isinstance(response, str)}")
    print(f"Length: {len(response)} characters")

    print(f"\nSummary response type: {type(summary).__name__}")
    print(f"Is string: {isinstance(summary, str)}")
    print(f"Length: {len(summary)} characters")

    return assistant


if __name__ == "__main__":
    # Run the main function
    main()
