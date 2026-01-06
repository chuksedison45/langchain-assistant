import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the application."""
    
    # AWS Configuration
    AWS_PROFILE: str = os.getenv("AWS_PROFILE", "default")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Bedrock Configuration
    MODEL_ID: str = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Model configuration validation
    SUPPORTED_MODELS = {
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-v2:1",
        "anthropic.claude-v2:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
    }
    
    @classmethod
    def validate_model_id(cls, model_id: Optional[str] = None) -> str:
        """Validate the model ID or use default."""
        model = model_id or cls.MODEL_ID
        if model not in cls.SUPPORTED_MODELS:
            print(f"Warning: Model {model} may not be supported. Using anyway.")
        return model
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("Current Configuration:")
        print(f"  AWS Profile: {cls.AWS_PROFILE}")
        print(f"  AWS Region: {cls.AWS_REGION}")
        print(f"  Model ID: {cls.MODEL_ID}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Max Tokens: {cls.MAX_TOKENS}")

# Create a config instance
config = Config()