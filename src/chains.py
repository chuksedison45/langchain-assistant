"""
Chains module for building complex LangChain chains.
This module extends the basic chain functionality with more complex chains.
"""

from typing import Optional, Dict, Any, List, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from src.chain import AssistantChain
from src.prompts import PromptFactory
from src.bedrock_client import BedrockClient
from src.memory import ConversationBuffer
from src.tools import ToolRegistry


class AdvancedChainBuilder:
    """Advanced chain builder with support for complex chain structures."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """Initialize with optional custom Bedrock client."""
        self.client = bedrock_client or BedrockClient()
        self.prompt_factory = PromptFactory()
        self.basic_chain_builder = AssistantChain(self.client)
        self.tool_registry = ToolRegistry()
        self.memory_store = ConversationBuffer()

    def create_conversational_chain(
        self,
        task: str = "assistant",
        memory_enabled: bool = True,
        max_history: int = 10,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a chain with conversation memory.

        Args:
            task: The task name
            memory_enabled: Whether to enable conversation memory
            max_history: Maximum conversation history to retain
            **kwargs: Additional chain parameters

        Returns:
            RunnableSequence with memory support
        """
        # Create the basic chain
        chain = self.basic_chain_builder.create_chain(task=task, **kwargs)

        if not memory_enabled:
            return chain

        # Wrap chain with memory
        def chain_with_memory(inputs: Dict[str, Any]) -> str:
            # Get conversation context from memory
            conversation_id = inputs.get("conversation_id", "default")
            history = self.memory_store.get_history(conversation_id, max_history)

            # Add history to inputs
            inputs_with_memory = inputs.copy()
            if history:
                inputs_with_memory["history"] = "\n".join(history)

            # Get response
            response = chain.invoke(inputs_with_memory)

            # Store in memory
            self.memory_store.add_message(
                conversation_id,
                {"role": "user", "content": inputs.get("message", "")}
            )
            self.memory_store.add_message(
                conversation_id,
                {"role": "assistant", "content": response}
            )

            return response

        return RunnableSequence(
            RunnablePassthrough(),
            chain_with_memory
        )

    def create_tool_calling_chain(
        self,
        tools: Optional[List[BaseTool]] = None,
        tool_names: Optional[List[str]] = None,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a chain that can call tools.

        Args:
            tools: List of LangChain tools
            tool_names: List of tool names to use from registry
            **kwargs: Additional chain parameters

        Returns:
            RunnableSequence with tool calling capability
        """
        # Get tools
        if tools is None:
            if tool_names is None:
                tool_names = ["calculator", "web_search", "time"]
            tools = self.tool_registry.get_tools(tool_names)

        # Create model
        model = self.client.create_chat_model(**kwargs)

        # Create prompt for tool calling
        prompt = self.prompt_factory.get_prompt_template("assistant")

        # Create chain with tool calling
        # Note: This is a simplified version; real tool calling requires agent setup
        chain = prompt | model | StrOutputParser()

        return chain

    def create_sequential_chain(
        self,
        chain_steps: List[Dict[str, Any]],
        **kwargs
    ) -> RunnableSequence:
        """
        Create a sequential chain of multiple steps.

        Args:
            chain_steps: List of chain configurations for each step
            **kwargs: Additional parameters

        Returns:
            Sequential chain
        """
        chains = []

        for step_config in chain_steps:
            task = step_config.get("task", "assistant")
            step_chain = self.basic_chain_builder.create_chain(
                task=task,
                **{**kwargs, **step_config}
            )
            chains.append(step_chain)

        # Create sequential chain
        def run_sequential(input_data: Dict[str, Any]) -> str:
            result = input_data
            for chain in chains:
                if isinstance(result, dict):
                    result = chain.invoke(result)
                else:
                    result = chain.invoke({"message": str(result)})
            return result

        return RunnableSequence(RunnablePassthrough(), run_sequential)

    def create_conditional_chain(
        self,
        condition_func: Callable[[Dict[str, Any]], str],
        chains_map: Dict[str, RunnableSequence],
        default_chain: Optional[RunnableSequence] = None,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a conditional chain that routes to different chains.

        Args:
            condition_func: Function that returns which chain to use
            chains_map: Dictionary mapping condition values to chains
            default_chain: Default chain if condition not in map
            **kwargs: Additional parameters

        Returns:
            Conditional chain
        """
        def conditional_routing(inputs: Dict[str, Any]) -> str:
            condition = condition_func(inputs)

            if condition in chains_map:
                chain = chains_map[condition]
            elif default_chain is not None:
                chain = default_chain
            else:
                raise ValueError(f"No chain for condition: {condition}")

            return chain.invoke(inputs)

        return RunnableSequence(RunnablePassthrough(), conditional_routing)

    def create_summarization_pipeline(
        self,
        extract_keywords: bool = False,
        generate_title: bool = False,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a multi-step summarization pipeline.

        Args:
            extract_keywords: Whether to extract keywords
            generate_title: Whether to generate a title
            **kwargs: Additional parameters

        Returns:
            Summarization pipeline chain
        """
        steps = []

        # Step 1: Initial summarization
        steps.append({
            "task": "summarizer",
            "length": "detailed"
        })

        # Step 2: Refine summary (optional)
        if extract_keywords or generate_title:
            steps.append({
                "task": "analyst",
                "focus": "text_analysis"
            })

        return self.create_sequential_chain(steps, **kwargs)

    def list_available_chains(self) -> Dict[str, str]:
        """List all available chain types."""
        return {
            "conversational": "Chain with conversation memory",
            "tool_calling": "Chain that can call tools",
            "sequential": "Chain of multiple sequential steps",
            "conditional": "Chain that routes based on conditions",
            "summarization_pipeline": "Multi-step summarization pipeline",
            "translation_chain": "Chain specialized for translation",
            "code_review_chain": "Chain for code review and analysis",
        }

    def get_chain_builder_info(self) -> Dict[str, Any]:
        """Get information about the chain builder."""
        return {
            "available_chains": self.list_available_chains(),
            "tools_registered": self.tool_registry.list_tools(),
            "memory_enabled": True,
            "max_conversations": self.memory_store.get_conversation_count(),
        }


class TranslationChain:
    """Specialized chain for translation tasks."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        self.client = bedrock_client or BedrockClient()
        self.prompt_factory = PromptFactory()

    def create_multi_step_translation_chain(
        self,
        preserve_formatting: bool = True,
        verify_translation: bool = False,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a multi-step translation chain.

        Args:
            preserve_formatting: Whether to preserve original formatting
            verify_translation: Whether to add verification step
            **kwargs: Additional parameters

        Returns:
            Multi-step translation chain
        """
        from src.chain import AssistantChain
        chain_builder = AssistantChain(self.client)

        # Step 1: Translation
        translate_chain = chain_builder.create_chain(
            task="translator",
            **kwargs
        )

        if not verify_translation:
            return translate_chain

        # Step 2: Verification (back-translation and comparison)
        def verify_translation_step(inputs: Dict[str, Any]) -> str:
            # Get original text and translation
            original_text = inputs.get("text", "")
            source_lang = inputs.get("source_language", "auto")
            target_lang = inputs.get("target_language", "English")

            # Translate
            translation = translate_chain.invoke(inputs)

            # Back-translate for verification
            back_translate_inputs = {
                "text": translation,
                "source_language": target_lang,
                "target_language": source_lang,
                "context": "Back translation for verification"
            }

            back_translation = translate_chain.invoke(back_translate_inputs)

            # Compare with original
            compare_prompt = ChatPromptTemplate.from_messages([
                ("system", "Compare two texts and report if the meaning is preserved."),
                ("human", f"""Original: {original_text}\nBack-translated:
                 {back_translation}\n\nIs the meaning preserved?""")
            ])

            model = self.client.create_chat_model(**kwargs)
            verification_chain = compare_prompt | model | StrOutputParser()
            verification = verification_chain.invoke({})

            return f"Translation: {translation}\n\nVerification: {verification}"

        return RunnableSequence(RunnablePassthrough(), verify_translation_step)


class CodeReviewChain:
    """Specialized chain for code review and analysis."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        self.client = bedrock_client or BedrockClient()
        self.prompt_factory = PromptFactory()

    def create_code_review_chain(
        self,
        language: str = "Python",
        review_aspects: List[str] = None,
        **kwargs
    ) -> RunnableSequence:
        """
        Create a chain for code review.

        Args:
            language: Programming language
            review_aspects: Aspects to review (syntax, style, security, etc.)
            **kwargs: Additional parameters

        Returns:
            Code review chain
        """
        if review_aspects is None:
            review_aspects = ["syntax", "style", "security", "performance",
                              "best_practices"]

        system_prompt = f"""You are an expert code reviewer for {language}.

Review the following code and provide feedback on:
{', '.join(review_aspects)}

Provide your review in this format:
1. **Overall Assessment**: [Brief summary]
2. **Specific Issues**: [List of issues found]
3. **Suggestions for Improvement**: [Specific suggestions]
4. **Security Considerations**: [Any security issues]
5. **Performance Notes**: [Performance implications]

Be thorough but constructive in your feedback."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Code to review:\n\n{code}")
        ])

        model = self.client.create_chat_model(**kwargs)
        return prompt | model | StrOutputParser()

    def create_code_explanation_chain(self, **kwargs) -> RunnableSequence:
        """Create a chain for explaining code."""
        system_prompt = """You are an expert programming educator.

Explain the provided code in detail, covering:
1. What the code does
2. How it works (step by step)
3. Key algorithms or patterns used
4. Time and space complexity (if applicable)
5. Potential edge cases
6. Alternative approaches

Make your explanation clear and accessible to developers of all levels."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Code to explain:\n\n{code}")
        ])

        model = self.client.create_chat_model(**kwargs)
        return prompt | model | StrOutputParser()
