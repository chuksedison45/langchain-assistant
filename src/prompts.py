"""
Prompt templates for different AI tasks.
This module contains reusable prompt templates for various use cases.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class PromptFactory:
    """Factory class for creating and managing prompt templates."""

    SUPPORTED_TASKS = {
        "assistant": "Multilingual general assistant",
        "summarizer": "Text summarization",
        "translator": "Text translation",
        "coder": "Code generation and explanation",
        "analyst": "Data analysis and insights",
        "creative": "Creative writing and brainstorming",
    }

    @classmethod
    def list_tasks(cls):
        """List all supported tasks and their descriptions."""
        print("Available Prompt Tasks:")
        print("-" * 40)
        for task, description in cls.SUPPORTED_TASKS.items():
            print(f"  {task:15} - {description}")
        return list(cls.SUPPORTED_TASKS.keys())

    @staticmethod
    def create_assistant_prompt(include_examples: bool = False) -> ChatPromptTemplate:
        """
        Create a multilingual assistant prompt template.

        Args:
            include_examples: Whether to include few-shot examples

        Returns:
            ChatPromptTemplate configured for assistant tasks
        """
        system_template = """You are a helpful, accurate, and concise AI assistant.

Your responsibilities:
1. Answer questions accurately based on your knowledge
2. Respond in the specified language: {language}
3. If you're unsure about something, acknowledge it
4. Format responses clearly with appropriate structure
5. Maintain a professional and helpful tone

Guidelines:
- For technical questions, include relevant details and context
- For creative tasks, be imaginative but coherent
- For factual questions, prioritize accuracy over brevity
- For complex topics, break down information into digestible parts

Current response language: {language}"""

        if include_examples:
            system_template += """
Examples of good responses:
Human: What is Python?
Assistant: Python is a high-level programming language known for its readability
and versatility. It supports multiple programming paradigms including object-oriented,
imperative, and functional programming.

Human: Explain quantum computing simply
Assistant: Quantum computing uses quantum bits (qubits) that can exist in multiple
states simultaneously. This allows quantum computers to solve certain
problems much faster than classical computers.
"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{message}"),
        ]

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def create_summarizer_prompt() -> ChatPromptTemplate:
        """
        Create a text summarization prompt template.

        Args:
            length: Desired summary length - "brief", "medium", or "detailed"

        Returns:
            ChatPromptTemplate configured for summarization
        """
        system_template = """You are a professional text summarizer.

Your task is to create clear, accurate summaries based on the provided text.

Summary length: {length}

Length guidelines:
- "brief": 1-2 sentences, key points only
- "medium": 3-5 sentences, main ideas with context
- "detailed": Multiple paragraphs, comprehensive coverage

Summary requirements:
1. Capture the main ideas and key points
2. Maintain the original meaning and context
3. Remove redundant information
4. Use clear, concise language
5. Preserve important technical terms and names
6. Do not add information not present in the original
7. Do not include personal opinions or commentary

Output format:
Start with: "Summary ({length}):"
Then provide the summary."""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("Text to summarize:\n\n{text}"),
        ]

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def create_translator_prompt() -> ChatPromptTemplate:
        """Create a text translation prompt template."""
        system_template = """You are a professional translator.

Translate the provided text from {source_language} to {target_language}.

Translation requirements:
1. Maintain the original meaning and intent
2. Adapt idioms and cultural references appropriately
3. Preserve technical terms (unless there's a standard translation)
4. Maintain the original tone (formal, informal, technical, etc.)
5. Ensure grammatical correctness in the target language
6. If the source text is ambiguous, make a reasonable interpretation

Additional context: {context}

Provide only the translation, no additional commentary."""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("Text to translate:\n\n{text}"),
        ]

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def create_coder_prompt() -> ChatPromptTemplate:
        """Create a coding assistant prompt template."""
        system_template = """You are an expert software developer and coding assistant.

You help with:
1. Writing clean, efficient code
2. Debugging and fixing errors
3. Explaining code concepts
4. Code review and optimization
5. Algorithm design and analysis

Programming language: {language}
Task type: {task_type} (implementation, explanation, debug, review)

Guidelines:
- Provide complete, runnable code when appropriate
- Include comments for complex logic
- Explain your approach and reasoning
- Consider edge cases and error handling
- Follow best practices and style guides
- Optimize for readability and maintainability

Additional requirements: {requirements}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{message}"),
        ]

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def create_analyst_prompt() -> ChatPromptTemplate:
        """Create a data analysis prompt template."""
        system_template = """You are a data analyst and insights generator.

Your task is to analyze the provided data or information and extract meaningful insights.

Analysis focus: {focus}
Audience: {audience}

Analysis guidelines:
1. Identify key patterns, trends, and outliers
2. Provide data-driven insights, not opinions
3. Use appropriate metrics and measurements
4. Consider context and limitations
5. Present findings clearly and concisely
6. Suggest actionable recommendations when appropriate
7. Use visual descriptions when helpful (tables, charts, etc.)

Data provided:
{data}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        return ChatPromptTemplate.from_messages(messages)

    @classmethod
    def get_prompt_template(cls, task: str, **kwargs) -> ChatPromptTemplate:
        """
        Get the appropriate prompt template based on task name.

        Args:
            task: The task name (assistant, summarizer, translator, etc.)
            **kwargs: Additional parameters for specific prompts

        Returns:
            Configured ChatPromptTemplate for the task

        Raises:
            ValueError: If task is not supported
        """
        task = task.lower()

        if task not in cls.SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' not supported. "
                f"Available tasks: {list(cls.SUPPORTED_TASKS.keys())}"
            )

        prompt_creators = {
            "assistant": cls.create_assistant_prompt,
            "summarizer": cls.create_summarizer_prompt,
            "translator": cls.create_translator_prompt,
            "coder": cls.create_coder_prompt,
            "analyst": cls.create_analyst_prompt,
            "creative": cls.create_assistant_prompt,  # Reuse assistant for creative
        }

        creator = prompt_creators[task]

        # Filter kwargs to only include those expected by the creator
        import inspect

        creator_params = inspect.signature(creator).parameters

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in creator_params}

        return creator(**filtered_kwargs)

    @classmethod
    def get_task_input_variables(cls, task: str) -> list:
        """
        Get the input variables required for a specific task.

        Args:
            task: The task name

        Returns:
            List of required input variable names
        """
        task = task.lower()

        task_variables = {
            "assistant": ["language", "message"],
            "summarizer": ["text", "length"],
            "translator": ["text", "source_language", "target_language", "context"],
            "coder": ["language", "task_type", "requirements", "message"],
            "analyst": ["data", "focus", "audience", "question"],
            "creative": ["language", "message"],
        }

        return task_variables.get(task, ["message"])

    @classmethod
    def create_dynamic_prompt(
        cls, task_description: str, custom_instructions: str = "", examples: list = None
    ) -> ChatPromptTemplate:
        """
        Create a dynamic prompt based on task description.

        Args:
            task_description: Description of what the AI should do
            custom_instructions: Additional specific instructions
            examples: List of example input/output pairs

        Returns:
            Custom ChatPromptTemplate
        """
        system_template = f"""You are an AI assistant specialized for a specific task.

Task Description:
{task_description}

{custom_instructions}

Guidelines:
1. Focus strictly on the task described above
2. Follow any specific formatting requirements
3. Ask for clarification if the request is ambiguous
4. Provide accurate and complete responses
5. If the task involves multiple steps, break them down clearly"""

        if examples:
            system_template += "\n\nExamples:\n"
            for i, (input_ex, output_ex) in enumerate(examples, 1):
                system_template += (
                    f"\nExample {i}:\nInput: {input_ex}\nOutput: {output_ex}\n"
                )

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]

        return ChatPromptTemplate.from_messages(messages)
