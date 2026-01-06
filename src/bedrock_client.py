import boto3
from botocore.config import Config as BotoConfig
from typing import Optional
from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import config

class BedrockClient:
    """AWS Bedrock client manager."""
    
    def __init__(self, region_name: Optional[str] = None, profile_name: Optional[str] = None):
        """Initialize Bedrock client with configuration."""
        self.region_name = region_name or config.AWS_REGION
        self.profile_name = profile_name or config.AWS_PROFILE
        
        # Create boto3 session with profile
        self.session = boto3.Session(
            profile_name=self.profile_name,
            region_name=self.region_name
        )
        
        # Create Bedrock runtime client
        self.client = self.session.client(
            service_name='bedrock-runtime',
            config=BotoConfig(
                retries={
                    'max_attempts': 5,
                    'mode': 'standard'
                }
            )
        )
        
        print(f"✓ Bedrock client initialized for region: {self.region_name}")
    
    def create_chat_model(
        self, 
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> BaseChatModel:
        """Create a LangChain ChatBedrock instance."""
        
        # Validate and get model ID
        validated_model_id = config.validate_model_id(model_id)
        
        # Get configuration values
        temp = temperature if temperature is not None else config.TEMPERATURE
        tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        
        try:
            # Create ChatBedrock instance
            chat_model = ChatBedrock(
                client=self.client,
                model_id=validated_model_id,
                model_kwargs={
                    "temperature": temp,
                    "max_tokens": tokens,
                },
                streaming=False,  # Set to True for streaming responses
            )
            
            print(f"✓ Chat model created: {validated_model_id}")
            print(f"  Temperature: {temp}, Max Tokens: {tokens}")
            
            return chat_model
            
        except Exception as e:
            print(f"✗ Error creating chat model: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check if the model is available in your region")
            print("2. Verify AWS credentials are configured")
            print("3. Ensure Bedrock access is enabled for the model")
            print("4. Check your AWS permissions (bedrock:InvokeModel)")
            raise
    
    def list_available_models(self):
        """List all available Bedrock models."""
        try:
            bedrock_client = self.session.client('bedrock')
            response = bedrock_client.list_foundation_models()
            
            print("\nAvailable Foundation Models:")
            print("-" * 50)
            
            models_by_provider = {}
            for model in response['modelSummaries']:
                provider = model['providerName']
                if provider not in models_by_provider:
                    models_by_provider[provider] = []
                models_by_provider[provider].append(model['modelId'])
            
            for provider, models in models_by_provider.items():
                print(f"\n{provider}:")
                for model in sorted(models):
                    print(f"  - {model}")
            
            return models_by_provider
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return {}