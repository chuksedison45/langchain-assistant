#!/usr/bin/env python3
"""
Example usage of multiple prompt templates.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import LangChainAssistant

def example_workflow():
    """Example workflow showing different prompt templates in action."""
    
    print("📚 Example: Using Multiple Prompt Templates")
    print("="*60)
    
    # Initialize assistant
    assistant = LangChainAssistant(verbose=True)
    
    print("\n1. Using the Assistant Prompt")
    print("-"*40)
    assistant.set_task("assistant")
    response = assistant.chat(
        "Explain the concept of neural networks",
        language="English"
    )
    
    print("\n2. Switching to Summarizer")
    print("-"*40)
    assistant.set_task("summarizer")
    
    long_text = """LangChain is a powerful framework designed to simplify the development 
    of applications using large language models (LLMs). It provides a standard interface 
    for chains, lots of integrations with other tools, and end-to-end chains for common 
    applications. This makes it easier to build complex applications that can leverage 
    the power of LLMs while managing context, memory, and tool use effectively."""
    
    response = assistant.summarize(long_text, length="brief")
    
    print("\n3. Switching to Translator")
    print("-"*40)
    assistant.set_task("translator")
    
    response = assistant.translate(
        "Hello, welcome to our AI assistant demonstration!",
        source_language="English",
        target_language="Spanish",
        context="Presentation introduction"
    )
    
    print("\n4. Switching to Code Assistant")
    print("-"*40)
    assistant.set_task("coder")
    
    response = assistant.code(
        "Create a function that sorts a list of dictionaries by a specific key",
        language="Python",
        task_type="implementation",
        requirements="Include type hints and error handling"
    )
    
    print("\n5. Task Information")
    print("-"*40)
    for task in ["assistant", "summarizer", "translator", "coder"]:
        info = assistant.get_task_info(task)
        print(f"\n{task.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Inputs needed: {info['required_inputs']}")
    
    print("\n" + "="*60)
    print("✅ Example workflow completed!")
    print("="*60)
    
    # Show history
    history = assistant.get_interaction_history()
    print(f"\nTotal interactions: {len(history)}")
    
    for i, entry in enumerate(history):
        print(f"\n{i+1}. [{entry['task']}]")
        print(f"   Time: {entry['response_time']:.2f}s")
        print(f"   Response preview: {entry['response'][:50]}...")

if __name__ == "__main__":
    example_workflow()