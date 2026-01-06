from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from src.bedrock_client import BedrockClient
from src.config import config

class AssistantChain:
    """Build and manage LangChain chains for the assistant."""
    
    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """Initialize with optional custom Bedrock client."""
        self.client = bedrock_client or BedrockClient()
        self.chain_cache: Dict[str, RunnableSequence] = {}
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template for a multilingual assistant."""
        
        system_prompt = """You are a helpful AI assistant. Your task is to:
1. Respond to the user's query accurately and helpfully
2. Use the specified language for your response
3. If the query is unclear, ask for clarification
4. Provide concise but complete answers
5. Format responses in a readable way

Current language: {language}

Please respond in {language}."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{message}")
        ])
    
    def create_chain(
        self, 
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> RunnableSequence:
        """Create a LangChain chain using LCEL."""
        
        # Create model
        model = self.client.create_chat_model(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template
        prompt = self.create_prompt_template()
        
        # Build chain using LCEL (LangChain Expression Language)
        chain = prompt | model | StrOutputParser()
        
        print("✓ LangChain pipeline created successfully")
        print(f"  Components: PromptTemplate → ChatBedrock → StrOutputParser")
        
        return chain
    
    def get_cached_chain(
        self, 
        cache_key: str = "default",
        **kwargs
    ) -> RunnableSequence:
        """Get a cached chain or create a new one."""
        if cache_key not in self.chain_cache:
            self.chain_cache[cache_key] = self.create_chain(**kwargs)
        return self.chain_cache[cache_key]
    
    def create_multilingual_chain(self) -> RunnableSequence:
        """Create a specialized chain for multilingual responses."""
        return self.create_chain()