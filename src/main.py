from typing import Optional, Dict, Any
import time
from datetime import datetime

from src.chain import AssistantChain
from src.config import config
from src.bedrock_client import BedrockClient

class LangChainAssistant:
    """Main application class for the LangChain assistant."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the assistant."""
        self.verbose = verbose
        self.client = BedrockClient()
        self.chain_builder = AssistantChain(self.client)
        self.chain = None
        self.interaction_history = []
        
        if verbose:
            config.print_config()
            print("\n" + "="*50)
            print("LangChain Assistant Initialized")
            print("="*50)
    
    def initialize_chain(self, **kwargs):
        """Initialize or reinitialize the chain with custom parameters."""
        self.chain = self.chain_builder.create_chain(**kwargs)
        return self
    
    def chat(
        self, 
        message: str, 
        language: str = "English",
        use_streaming: bool = False,
        **kwargs
    ) -> str:
        """Send a message to the assistant and get a response."""
        
        # Initialize chain if not already done
        if self.chain is None:
            self.initialize_chain()
        
        # Prepare inputs
        inputs = {
            "language": language,
            "message": message,
            **kwargs
        }
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Query: {message}")
            print(f"Language: {language}")
            print(f"{'='*50}\n")
        
        try:
            # Measure response time
            start_time = time.time()
            
            # Invoke the chain
            response = self.chain.invoke(inputs)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log interaction
            self.interaction_history.append({
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "message": message,
                "response": response,
                "response_time": response_time
            })
            
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Response (in {language}):")
                print(f"{'='*50}")
                print(response)
                print(f"{'='*50}")
                print(f"Response time: {response_time:.2f} seconds")
                print(f"{'='*50}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting response: {e}"
            print(f"\n✗ {error_msg}")
            return error_msg
    
    def batch_chat(
        self, 
        conversations: list,
        **kwargs
    ) -> list:
        """Process multiple conversations in sequence."""
        results = []
        
        for i, conv in enumerate(conversations):
            print(f"\n[{i+1}/{len(conversations)}] Processing conversation...")
            
            if isinstance(conv, dict):
                message = conv.get("message", "")
                language = conv.get("language", "English")
            else:
                message = conv
                language = "English"
            
            response = self.chat(
                message=message,
                language=language,
                verbose=False,
                **kwargs
            )
            
            results.append({
                "message": message,
                "language": language,
                "response": response
            })
        
        return results
    
    def get_interaction_history(self):
        """Return the interaction history."""
        return self.interaction_history
    
    def clear_history(self):
        """Clear the interaction history."""
        self.interaction_history = []
        print("✓ Interaction history cleared")

def main():
    """Main entry point for the application."""
    
    # Create assistant instance
    assistant = LangChainAssistant(verbose=True)
    
    # Example conversations
    conversations = [
        {
            "language": "English",
            "message": "Explain the concept of machine learning in simple terms."
        },
        {
            "language": "Spanish",
            "message": "¿Cuáles son los beneficios de la inteligencia artificial?"
        },
        {
            "language": "French",
            "message": "Parlez-moi de l'apprentissage en profondeur."
        },
        {
            "language": "German", 
            "message": "Was ist natürliche Sprachverarbeitung?"
        }
    ]
    
    # Process conversations
    print("\n" + "="*50)
    print("Starting Multilingual Conversations")
    print("="*50)
    
    results = assistant.batch_chat(conversations)
    
    # Print summary
    print("\n" + "="*50)
    print("Conversation Summary")
    print("="*50)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. [{result['language']}]")
        print(f"   Q: {result['message'][:80]}...")
        print(f"   A: {result['response'][:100]}...")

def chat_interactive():
    """Interactive chat mode."""
    
    assistant = LangChainAssistant(verbose=True)
    
    print("\n" + "="*50)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end")
    print("Type 'clear' to clear history")
    print("Type 'history' to view conversation history")
    print("="*50)
    
    while True:
        try:
            # Get language preference
            language = input("\nEnter language (default: English): ").strip()
            if not language:
                language = "English"
            
            if language.lower() in ['quit', 'exit']:
                break
            
            # Get message
            message = input(f"\n[{language}] You: ").strip()
            
            if not message:
                continue
            
            if message.lower() in ['quit', 'exit']:
                break
            elif message.lower() == 'clear':
                assistant.clear_history()
                continue
            elif message.lower() == 'history':
                history = assistant.get_interaction_history()
                if history:
                    print("\nConversation History:")
                    for i, entry in enumerate(history):
                        print(f"\n{i+1}. [{entry['language']}] {entry['timestamp']}")
                        print(f"   Q: {entry['message'][:50]}...")
                        print(f"   A: {entry['response'][:50]}...")
                else:
                    print("No conversation history.")
                continue
            
            # Get response
            assistant.chat(message, language)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment to run interactive mode
    # chat_interactive()