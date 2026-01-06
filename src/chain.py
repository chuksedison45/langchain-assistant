from typing import Optional, Dict, Any, Literal, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.bedrock_client import BedrockClient
from src.config import config
from src.prompts import PromptFactory

class AssistantChain:
    """Build and manage LangChain chains for various tasks."""
    
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
        **prompt_kwargs
    ) -> RunnableSequence:
        """
        Create a LangChain chain for a specific task using LCEL.
        
        Args:
            task: The task name (assistant, summarizer, translator, etc.)
            model_id: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **prompt_kwargs: Additional parameters for the prompt
            
        Returns:
            RunnableSequence chain
        """
        # Validate task
        if task not in self.prompt_factory.SUPPORTED_TASKS:
            available = list(self.prompt_factory.SUPPORTED_TASKS.keys())
            raise ValueError(
                f"Task '{task}' not supported. Available tasks: {available}"
            )
        
        # Create model
        model = self.client.create_chat_model(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Get appropriate prompt template
        prompt = self.prompt_factory.get_prompt_template(task, **prompt_kwargs)
        
        # Build chain using LCEL
        chain = prompt | model | StrOutputParser()
        
        task_description = self.prompt_factory.SUPPORTED_TASKS[task]
        print(f"✓ Chain created for task: {task} ({task_description})")
        
        return chain
    
    def create_dynamic_chain(
        self,
        task_description: str,
        custom_instructions: str = "",
        examples: list = None,
        **kwargs
    ) -> RunnableSequence:
        """Create a chain with a dynamically generated prompt."""
        
        # Create model
        model = self.client.create_chat_model(
            model_id=kwargs.get('model_id'),
            temperature=kwargs.get('temperature'),
            max_tokens=kwargs.get('max_tokens')
        )
        
        # Create dynamic prompt
        prompt = self.prompt_factory.create_dynamic_prompt(
            task_description,
            custom_instructions,
            examples
        )
        
        # Build chain
        chain = prompt | model | StrOutputParser()
        
        print("✓ Dynamic chain created with custom prompt")
        
        return chain
    
    def get_chain(
        self,
        task: str = "assistant",
        cache_key: Optional[str] = None,
        **kwargs
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
    
    def list_available_chains(self):
        """List all available chain types."""
        return self.prompt_factory.list_tasks()
    
    def create_summarizer_chain(
        self,
        length: Literal["brief", "medium", "detailed"] = "medium",
        **kwargs
    ) -> RunnableSequence:
        """Create a specialized summarizer chain."""
        return self.create_chain(
            task="summarizer",
            length=length,
            **kwargs
        )
    
    def create_translator_chain(
        self,
        source_language: str = "auto",
        target_language: str = "English",
        context: str = "",
        **kwargs
    ) -> RunnableSequence:
        """Create a specialized translator chain."""
        return self.create_chain(
            task="translator",
            source_language=source_language,
            target_language=target_language,
            context=context,
            **kwargs
        )
    
    def create_coder_chain(
        self,
        language: str = "Python",
        task_type: str = "implementation",
        requirements: str = "",
        **kwargs
    ) -> RunnableSequence:
        """Create a specialized coder chain."""
        return self.create_chain(
            task="coder",
            language=language,
            task_type=task_type,
            requirements=requirements,
            **kwargs
        )