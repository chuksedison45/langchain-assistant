from typing import Optional, Dict, Any, Literal
import time
from datetime import datetime

from src.chain import AssistantChain
from src.config import config
from src.bedrock_client import BedrockClient
from src.prompts import PromptFactory

class LangChainAssistant:
    """Main application class for the LangChain assistant with multiple prompt templates."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the assistant."""
        self.verbose = verbose
        self.client = BedrockClient()
        self.chain_builder = AssistantChain(self.client)
        self.prompt_factory = PromptFactory()
        
        # Current chain and task
        self.current_task = "assistant"
        self.current_chain = None
        
        # Task-specific configurations
        self.task_configs = {
            "assistant": {"language": "English"},
            "summarizer": {"length": "medium"},
            "translator": {"source_language": "auto", "target_language": "English"},
            "coder": {"language": "Python", "task_type": "implementation"},
            "analyst": {"focus": "general", "audience": "general"},
        }
        
        self.interaction_history = []
        
        if verbose:
            config.print_config()
            print("\n" + "="*60)
            print("LangChain Assistant with Multiple Prompt Templates")
            print("="*60)
            self.prompt_factory.list_tasks()
    
    def set_task(
        self, 
        task: str, 
        **task_kwargs
    ) -> "LangChainAssistant":
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
            task=task,
            **task_kwargs
        )
        
        if self.verbose:
            task_desc = self.prompt_factory.SUPPORTED_TASKS[task]
            print(f"\n✓ Task set to: {task} ({task_desc})")
            print(f"  Required inputs: {input_vars}")
            if task_kwargs:
                print(f"  Configuration: {task_kwargs}")
        
        return self
    
    def process(
        self, 
        input_data: Dict[str, Any],
        task: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Process input using the specified or current task.
        
        Args:
            input_data: Dictionary of input variables for the prompt
            task: Optional task name (uses current task if not specified)
            **kwargs: Additional chain parameters
            
        Returns:
            AI response as string
        """
        # Use specified task or current task
        target_task = task or self.current_task
        
        # Get or create chain
        if task and task != self.current_task:
            chain = self.chain_builder.get_chain(task=task, **kwargs)
        else:
            chain = self.current_chain or self.chain_builder.get_chain(task=target_task, **kwargs)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {target_task}")
            print(f"Input: {input_data}")
            print(f"{'='*60}\n")
        
        try:
            # Measure response time
            start_time = time.time()
            
            # Invoke the chain
            response = chain.invoke(input_data)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log interaction
            self.interaction_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": target_task,
                "input": input_data,
                "response": response,
                "response_time": response_time
            })
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Response:")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}")
                print(f"Response time: {response_time:.2f} seconds")
                print(f"{'='*60}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting response: {e}"
            print(f"\n✗ {error_msg}")
            return error_msg
    
    def chat(
        self, 
        message: str, 
        language: str = "English",
        **kwargs
    ) -> str:
        """Chat using assistant task."""
        return self.process(
            {"language": language, "message": message},
            task="assistant",
            **kwargs
        )
    
    def summarize(
        self,
        text: str,
        length: Literal["brief", "medium", "detailed"] = "medium",
        **kwargs
    ) -> str:
        """Summarize text with specified length."""
        return self.process(
            {"text": text, "length": length},
            task="summarizer",
            **kwargs
        )
    
    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "English",
        context: str = "",
        **kwargs
    ) -> str:
        """Translate text between languages."""
        return self.process(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "context": context
            },
            task="translator",
            **kwargs
        )
    
    def code(
        self,
        request: str,
        language: str = "Python",
        task_type: str = "implementation",
        requirements: str = "",
        **kwargs
    ) -> str:
        """Generate or explain code."""
        return self.process(
            {
                "message": request,
                "language": language,
                "task_type": task_type,
                "requirements": requirements
            },
            task="coder",
            **kwargs
        )
    
    def analyze(
        self,
        data: str,
        question: str,
        focus: str = "general",
        audience: str = "general",
        **kwargs
    ) -> str:
        """Analyze data and provide insights."""
        return self.process(
            {
                "data": data,
                "question": question,
                "focus": focus,
                "audience": audience
            },
            task="analyst",
            **kwargs
        )
    
    def get_task_info(self, task: str = None) -> Dict:
        """Get information about a specific task."""
        task = task or self.current_task
        return {
            "task": task,
            "description": self.prompt_factory.SUPPORTED_TASKS.get(task, "Unknown"),
            "required_inputs": self.prompt_factory.get_task_input_variables(task),
            "current_config": self.task_configs.get(task, {})
        }
    
    def get_interaction_history(self):
        """Return the interaction history."""
        return self.interaction_history
    
    def clear_history(self):
        """Clear the interaction history."""
        self.interaction_history = []
        print("✓ Interaction history cleared")

def demo_multiple_tasks():
    """Demonstrate multiple prompt templates."""
    
    print("\n" + "="*60)
    print("Demo: Multiple Prompt Templates")
    print("="*60)
    
    # Create assistant
    assistant = LangChainAssistant(verbose=True)
    
    # Test cases for different tasks
    test_cases = [
        {
            "task": "assistant",
            "name": "Multilingual Assistant",
            "input": {"language": "Spanish", "message": "¿Qué es la inteligencia artificial?"},
            "method": "chat",
            "args": ["¿Qué es la inteligencia artificial?", "Spanish"]
        },
        {
            "task": "summarizer",
            "name": "Text Summarizer",
            "input": {
                "text": """Artificial intelligence (AI) is intelligence demonstrated by machines, 
                as opposed to natural intelligence displayed by animals including humans. 
                Leading AI textbooks define the field as the study of intelligent agents: 
                any system that perceives its environment and takes actions that maximize 
                its chance of achieving its goals. Some popular accounts use the term 
                artificial intelligence to describe machines that mimic cognitive functions 
                that humans associate with the human mind, such as learning and problem 
                solving.""",
                "length": "brief"
            },
            "method": "summarize",
            "args": ["""Artificial intelligence (AI) is intelligence demonstrated by machines, 
                as opposed to natural intelligence displayed by animals including humans. 
                Leading AI textbooks define the field as the study of intelligent agents: 
                any system that perceives its environment and takes actions that maximize 
                its chance of achieving its goals. Some popular accounts use the term 
                artificial intelligence to describe machines that mimic cognitive functions 
                that humans associate with the human mind, such as learning and problem 
                solving.""", "brief"]
        },
        {
            "task": "translator",
            "name": "Text Translator",
            "input": {
                "text": "Hello, how are you today?",
                "source_language": "English",
                "target_language": "French",
                "context": "Casual conversation"
            },
            "method": "translate",
            "args": ["Hello, how are you today?", "English", "French", "Casual conversation"]
        },
        {
            "task": "coder",
            "name": "Code Assistant",
            "input": {
                "message": "Write a Python function to calculate Fibonacci numbers",
                "language": "Python",
                "task_type": "implementation",
                "requirements": "Include error handling and comments"
            },
            "method": "code",
            "args": ["Write a Python function to calculate Fibonacci numbers", "Python", "implementation", "Include error handling and comments"]
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test['name']}")
        print(f"Task: {test['task']}")
        print(f"{'='*60}")
        
        try:
            # Set task
            assistant.set_task(test['task'])
            
            # Process using appropriate method
            if test['method'] == 'chat':
                response = assistant.chat(*test['args'])
            elif test['method'] == 'summarize':
                response = assistant.summarize(*test['args'])
            elif test['method'] == 'translate':
                response = assistant.translate(*test['args'])
            elif test['method'] == 'code':
                response = assistant.code(*test['args'])
            else:
                response = assistant.process(test['input'], task=test['task'])
            
            results.append({
                "task": test['task'],
                "success": True,
                "response_length": len(response),
                "response_preview": response[:100] + "..." if len(response) > 100 else response
            })
            
            print(f"✓ Task completed successfully")
            
        except Exception as e:
            print(f"✗ Task failed: {e}")
            results.append({
                "task": test['task'],
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("Demo Summary")
    print(f"{'='*60}")
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status} {result['task']:15}")
        if result['success']:
            print(f"     Response: {result['response_preview']}")
    
    return results

def interactive_mode():
    """Interactive mode with task switching."""
    
    assistant = LangChainAssistant(verbose=True)
    
    print("\n" + "="*60)
    print("Interactive Mode with Multiple Tasks")
    print("="*60)
    print("Commands:")
    print("  task <name>        - Switch to a task (e.g., 'task summarizer')")
    print("  list               - List available tasks")
    print("  info               - Show current task info")
    print("  history            - Show interaction history")
    print("  clear              - Clear history")
    print("  quit/exit          - Exit")
    print("="*60)
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            elif command == 'list':
                assistant.prompt_factory.list_tasks()
                
            elif command == 'info':
                info = assistant.get_task_info()
                print(f"\nCurrent Task: {info['task']}")
                print(f"Description: {info['description']}")
                print(f"Required Inputs: {info['required_inputs']}")
                print(f"Configuration: {info['current_config']}")
                
            elif command == 'history':
                history = assistant.get_interaction_history()
                if history:
                    print(f"\nInteraction History ({len(history)} entries):")
                    for i, entry in enumerate(history[-5:]):  # Show last 5
                        print(f"\n{i+1}. [{entry['task']}] {entry['timestamp']}")
                        print(f"   Input: {str(entry['input'])[:50]}...")
                        print(f"   Response: {entry['response'][:50]}...")
                else:
                    print("No interaction history.")
                    
            elif command == 'clear':
                assistant.clear_history()
                
            elif command.startswith('task '):
                task_name = command[5:].strip()
                try:
                    assistant.set_task(task_name)
                except ValueError as e:
                    print(f"Error: {e}")
                    
            else:
                # Process as input for current task
                current_task = assistant.current_task
                task_info = assistant.get_task_info()
                
                print(f"\nCurrent task: {current_task}")
                print(f"Required inputs: {task_info['required_inputs']}")
                
                # Collect inputs based on task
                inputs = {}
                for input_var in task_info['required_inputs']:
                    value = input(f"  {input_var}: ").strip()
                    inputs[input_var] = value
                
                if inputs:
                    response = assistant.process(inputs)
                else:
                    print("No inputs provided.")
                    
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main entry point for the application - maintains backward compatibility."""
    
    # Create assistant instance
    assistant = LangChainAssistant(verbose=True)
    
    # Default to assistant mode for backward compatibility
    assistant.set_task("assistant")
    
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
    print("\n" + "="*60)
    print("Starting Multilingual Conversations (Backward Compatibility)")
    print("="*60)
    
    results = []
    
    for i, conv in enumerate(conversations):
        print(f"\n[{i+1}/{len(conversations)}] Processing conversation...")
        
        if isinstance(conv, dict):
            message = conv.get("message", "")
            language = conv.get("language", "English")
        else:
            message = conv
            language = "English"
        
        response = assistant.chat(
            message=message,
            language=language,
            verbose=False
        )
        
        results.append({
            "message": message,
            "language": language,
            "response": response
        })
    
    # Print summary
    print("\n" + "="*60)
    print("Conversation Summary")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. [{result['language']}]")
        print(f"   Q: {result['message'][:80]}...")
        print(f"   A: {result['response'][:100]}...")
    
    # Show that we can also use the new features
    print("\n" + "="*60)
    print("Demonstrating New Features (Part 3)")
    print("="*60)
    print("\nYou can now use multiple prompt templates:")
    print("  - assistant.chat() - For general conversation")
    print("  - assistant.summarize() - For text summarization")
    print("  - assistant.translate() - For translation")
    print("  - assistant.code() - For code generation")
    print("  - assistant.analyze() - For data analysis")
    
    # Quick demo of summarizer
    print("\nQuick summarizer demo:")
    sample_text = "LangChain is a framework for building applications with large language models. It provides tools and abstractions to make it easier to work with LLMs, handle context, and build complex chains of operations."
    
    print(f"\nOriginal text: {sample_text[:80]}...")
    summary = assistant.summarize(sample_text, length="brief", verbose=False)
    print(f"\nBrief summary: {summary}")
    
    return results

def chat_interactive():
    """Interactive chat mode with backward compatibility."""
    
    assistant = LangChainAssistant(verbose=True)
    
    print("\n" + "="*60)
    print("Interactive Chat Mode (Now with Task Switching)")
    print("Type 'quit' or 'exit' to end")
    print("Type 'task <name>' to switch tasks")
    print("Type 'list' to see available tasks")
    print("Type 'clear' to clear history")
    print("Type 'history' to view conversation history")
    print("="*60)
    
    while True:
        try:
            # Get language preference
            language = input("\nEnter language (default: English): ").strip()
            if not language:
                language = "English"
            
            if language.lower() in ['quit', 'exit']:
                break
            
            # Check for commands
            if language.lower() == 'list':
                assistant.prompt_factory.list_tasks()
                continue
            elif language.lower().startswith('task '):
                task_name = language[5:].strip()
                try:
                    assistant.set_task(task_name)
                except ValueError as e:
                    print(f"Error: {e}")
                continue
            elif language.lower() == 'clear':
                assistant.clear_history()
                continue
            elif language.lower() == 'history':
                history = assistant.get_interaction_history()
                if history:
                    print("\nConversation History:")
                    for i, entry in enumerate(history):
                        print(f"\n{i+1}. [{entry['task']}] {entry['timestamp']}")
                        print(f"   Q: {entry['input'].get('message', str(entry['input']))[:50]}...")
                        print(f"   A: {entry['response'][:50]}...")
                else:
                    print("No conversation history.")
                continue
            
            # Get message
            message = input(f"\n[{language}] You: ").strip()
            
            if not message:
                continue
            
            if message.lower() in ['quit', 'exit']:
                break
            
            # Get response
            assistant.chat(message, language)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    # Run the main function (backward compatible)
    main()
    
    # Uncomment to run interactive mode with task switching
    # chat_interactive()
    
    # Uncomment to run the multiple tasks demo
    # demo_multiple_tasks()
    
    # Uncomment to run the new interactive mode
    # interactive_mode()