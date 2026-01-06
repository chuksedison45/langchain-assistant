#!/usr/bin/env python3
"""
Test script for output parsing functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import LangChainAssistant

def test_output_parsing():
    """Test output parsing functionality."""
    
    print("="*70)
    print("Testing Output Parsing")
    print("="*70)
    
    assistant = LangChainAssistant(verbose=False)
    
    print("\n1. Testing chat() returns clean string:")
    response = assistant.chat("What is 2+2?", "English")
    
    if isinstance(response, str) and len(response) > 0:
        print(f"   ✓ PASS: Response is string, length: {len(response)}")
        print(f"   Preview: {response[:50]}...")
    else:
        print(f"   ✗ FAIL: Response is not a string or empty")
        print(f"   Type: {type(response)}, Length: {len(str(response))}")
    
    print("\n2. Testing summarizer() returns clean string:")
    
    sample_text = """Python is a high-level programming language known for its 
    simplicity and readability. It supports multiple programming paradigms 
    including object-oriented, imperative, and functional programming."""
    
    summary = assistant.summarize(sample_text, length="brief")
    
    if isinstance(summary, str) and len(summary) > 0:
        print(f"   ✓ PASS: Summary is string, length: {len(summary)}")
        print(f"   Preview: {summary[:50]}...")
    else:
        print(f"   ✗ FAIL: Summary is not a string or empty")
        print(f"   Type: {type(summary)}, Length: {len(str(summary))}")
    
    print("\n3. Testing both prompts work with output parser:")
    
    test_cases = [
        {
            "name": "Assistant prompt",
            "method": "chat",
            "args": ["Explain artificial intelligence", "English"],
        },
        {
            "name": "Summarizer prompt", 
            "method": "summarize",
            "args": [sample_text, "medium"],
        }
    ]
    
    for test in test_cases:
        print(f"\n   Testing {test['name']}:")
        
        if test['method'] == 'chat':
            response = assistant.chat(*test['args'])
        elif test['method'] == 'summarize':
            response = assistant.summarize(*test['args'])
        
        if isinstance(response, str):
            print(f"     ✓ Returns string, length: {len(response)}")
        else:
            print(f"     ✗ Does not return string")
    
    print("\n" + "="*70)
    print("Output parsing tests completed!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    print("Starting Output Parsing Tests...")
    test_output_parsing()
    print("\nAll tests completed!")