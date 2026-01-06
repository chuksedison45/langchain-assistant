#!/usr/bin/env python3
"""
Example demonstrating output parsing in action.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.main import LangChainAssistant


def demonstrate_output_parsing():
    """Show output parsing in action with different scenarios."""

    print("🔧 Example: Output Parsing in Action")
    print("=" * 60)

    # Initialize assistant
    assistant = LangChainAssistant(verbose=True)

    print("\n1. Testing that chat() returns clean string:")
    print("-" * 40)

    response = assistant.chat("What is artificial intelligence?", "English")

    print(f"\nResponse type: {type(response).__name__}")
    print(f"Is string: {isinstance(response, str)}")
    print(f"Response preview: {response[:100]}...")

    print("\n2. Testing summarizer with output parsing:")
    print("-" * 40)

    text_to_summarize = """LangChain is a framework designed to simplify the development 
    of applications using large language models (LLMs). It provides a standard interface 
    for chains, lots of integrations with other tools, and end-to-end chains for common 
    applications."""

    for length in ["brief", "medium", "detailed"]:
        print(f"\n{length.capitalize()} summary:")
        summary = assistant.summarize(text_to_summarize, length=length)
        print(f"Type: {type(summary).__name__}")
        print(f"Length: {len(summary)} characters")
        print(f"Preview: {summary[:100]}...")

    print("\n" + "=" * 60)
    print("✅ Output parsing example completed!")
    print("All responses are clean strings thanks to StrOutputParser")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_output_parsing()
