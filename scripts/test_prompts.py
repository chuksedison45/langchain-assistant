#!/usr/bin/env python3
"""
Test script for multiple prompt templates.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import LangChainAssistant

def test_all_prompts():
    """Test all available prompt templates."""
    
    print("="*70)
    print("Testing All Prompt Templates")
    print("="*70)
    
    assistant = LangChainAssistant(verbose=False)
    
    test_cases = [
        {
            "task": "assistant",
            "inputs": {"language": "English", "message": "What is machine learning?"},
            "description": "Basic assistant query"
        },
        {
            "task": "assistant",
            "inputs": {"language": "French", "message": "Explique l'apprentissage automatique."},
            "description": "Multilingual assistant"
        },
        {
            "task": "summarizer",
            "inputs": {
                "text": """The Python programming language is known for its simplicity and readability. 
                It supports multiple programming paradigms including object-oriented, imperative, 
                and functional programming. Python has a comprehensive standard library and 
                a large ecosystem of third-party packages.""",
                "length": "brief"
            },
            "description": "Brief text summary"
        },
        {
            "task": "summarizer", 
            "inputs": {
                "text": """Artificial intelligence is transforming industries worldwide. 
                From healthcare to finance, AI applications are improving efficiency and 
                enabling new capabilities. Machine learning, a subset of AI, allows systems 
                to learn from data without explicit programming.""",
                "length": "detailed"
            },
            "description": "Detailed text summary"
        },
        {
            "task": "translator",
            "inputs": {
                "text": "Good morning! How can I help you today?",
                "source_language": "English",
                "target_language": "Spanish",
                "context": "Customer service greeting"
            },
            "description": "Text translation"
        },
        {
            "task": "coder",
            "inputs": {
                "message": "Write a function to check if a string is a palindrome",
                "language": "Python",
                "task_type": "implementation",
                "requirements": "Include test cases and handle edge cases"
            },
            "description": "Code generation"
        },
        {
            "task": "analyst",
            "inputs": {
                "data": "Sales increased by 15% last quarter, but customer complaints rose by 5%",
                "question": "What insights can you draw from this data?",
                "focus": "business",
                "audience": "executives"
            },
            "description": "Data analysis"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}] {test['task'].upper()}: {test['description']}")
        print("-" * 60)
        
        try:
            # Set task
            assistant.set_task(test['task'])
            
            # Process
            response = assistant.process(test['inputs'], verbose=False)
            
            # Validate
            if response and len(response) > 10:
                status = "PASS"
                print(f"✓ Response received ({len(response)} characters)")
                print(f"  Preview: {response[:100]}...")
            else:
                status = "FAIL"
                print(f"✗ Empty or short response")
                
        except Exception as e:
            status = "ERROR"
            print(f"✗ Error: {e}")
            response = None
        
        results.append({
            "test": i+1,
            "task": test['task'],
            "description": test['description'],
            "status": status,
            "response_length": len(response) if response else 0
        })
    
    # Summary
    print(f"\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed} | Failed: {failed} | Errors: {errors}")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for result in results:
        symbol = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "⚠"
        print(f"{symbol} Test {result['test']:2} - {result['task']:12} {result['description']:30} "
              f"({result['response_length']} chars)")
    
    return all(r['status'] == 'PASS' for r in results)

def test_prompt_switching():
    """Test switching between different prompts dynamically."""
    
    print(f"\n" + "="*70)
    print("Testing Prompt Switching")
    print("="*70)
    
    assistant = LangChainAssistant(verbose=False)
    
    # Test switching without recreating assistant
    tasks_to_test = ["assistant", "summarizer", "translator", "coder"]
    
    for task in tasks_to_test:
        print(f"\nSwitching to task: {task}")
        
        try:
            assistant.set_task(task)
            task_info = assistant.get_task_info()
            
            print(f"  ✓ Successfully switched")
            print(f"    Description: {task_info['description']}")
            print(f"    Required inputs: {task_info['required_inputs']}")
            
        except Exception as e:
            print(f"  ✗ Failed to switch: {e}")
    
    return True

def test_custom_configurations():
    """Test prompt templates with custom configurations."""
    
    print(f"\n" + "="*70)
    print("Testing Custom Configurations")
    print("="*70)
    
    assistant = LangChainAssistant(verbose=False)
    
    # Test custom summarizer lengths
    sample_text = """LangChain is a framework for developing applications powered by language models. 
    It enables applications that are context-aware and reason about what to do based on the context."""
    
    lengths = ["brief", "medium", "detailed"]
    
    for length in lengths:
        print(f"\nSummarizing with length: {length}")
        
        response = assistant.summarize(sample_text, length=length, verbose=False)
        
        print(f"  Response length: {len(response)} characters")
        print(f"  Preview: {response[:80]}...")
    
    # Test translator with different contexts
    text_to_translate = "The weather is nice today."
    
    contexts = ["", "Casual conversation", "Weather report", "Poetic"]
    
    for context in contexts:
        print(f"\nTranslating with context: '{context}'")
        
        response = assistant.translate(
            text_to_translate,
            source_language="English",
            target_language="French",
            context=context,
            verbose=False
        )
        
        print(f"  Translation: {response}")
    
    return True

if __name__ == "__main__":
    print("Starting Prompt Template Tests...")
    
    # Run tests
    test_all_prompts()
    test_prompt_switching()
    test_custom_configurations()
    
    print(f"\n" + "="*70)
    print("All prompt template tests completed!")
    print("="*70)