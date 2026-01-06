#!/usr/bin/env python3
"""
Test script for the LangChain Assistant.
Tests the assistant with different languages and scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import LangChainAssistant

def test_basic_functionality():
    """Test basic functionality of the assistant."""
    
    print("="*60)
    print("Testing Basic LangChain Assistant")
    print("="*60)
    
    # Create assistant
    assistant = LangChainAssistant(verbose=True)
    
    # Test cases
    test_cases = [
        {
            "name": "English - Technical Question",
            "language": "English",
            "message": "What is LangChain and what are its main components?",
        },
        {
            "name": "Spanish - General Knowledge",
            "language": "Spanish",
            "message": "Explica la importancia de la inteligencia artificial en la medicina moderna.",
        },
        {
            "name": "French - Creative Task",
            "language": "French", 
            "message": "Écris un court poème sur la technologie et l'humanité.",
        },
        {
            "name": "German - Practical Advice",
            "language": "German",
            "message": "Gib mir 5 Tipps für besseres Zeitmanagement bei der Softwareentwicklung.",
        },
        {
            "name": "Italian - Cultural Question",
            "language": "Italian",
            "message": "Qual è l'impatto del Rinascimento sullo sviluppo scientifico europeo?",
        }
    ]
    
    # Run tests
    all_passed = True
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test['name']}")
        print(f"{'='*60}")
        
        try:
            response = assistant.chat(
                message=test['message'],
                language=test['language'],
                verbose=True
            )
            
            # Basic validation
            if response and len(response) > 10:
                print(f"✓ Test {i+1} PASSED")
                print(f"  Response length: {len(response)} characters")
            else:
                print(f"✗ Test {i+1} FAILED - Empty or too short response")
                all_passed = False
                
        except Exception as e:
            print(f"✗ Test {i+1} FAILED with error: {e}")
            all_passed = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    
    # Print interaction history
    history = assistant.get_interaction_history()
    print(f"\nTotal interactions: {len(history)}")
    
    total_time = sum(entry['response_time'] for entry in history)
    avg_time = total_time / len(history) if history else 0
    
    print(f"Average response time: {avg_time:.2f} seconds")
    
    return all_passed

def test_error_handling():
    """Test error handling scenarios."""
    
    print(f"\n{'='*60}")
    print("Testing Error Handling")
    print(f"{'='*60}")
    
    assistant = LangChainAssistant(verbose=False)
    
    # Test with invalid input
    try:
        response = assistant.chat(
            message="",
            language="English"
        )
        print("✓ Empty message handled gracefully")
    except Exception as e:
        print(f"✗ Error with empty message: {e}")
    
    # Test with very long message
    long_message = "test " * 1000
    try:
        response = assistant.chat(
            message=long_message,
            language="English"
        )
        print("✓ Long message handled")
    except Exception as e:
        print(f"✗ Error with long message: {e}")

def test_different_models():
    """Test with different Bedrock models (if available)."""
    
    print(f"\n{'='*60}")
    print("Testing Different Models")
    print(f"{'='*60}")
    
    models_to_test = [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-v2:1",
        # Add more models as needed
    ]
    
    for model_id in models_to_test:
        print(f"\nTesting model: {model_id}")
        
        try:
            # Create assistant with specific model
            assistant = LangChainAssistant(verbose=False)
            assistant.initialize_chain(model_id=model_id)
            
            response = assistant.chat(
                message="What is 2+2?",
                language="English",
                verbose=False
            )
            
            if "4" in response or "four" in response.lower():
                print(f"✓ Model {model_id} working correctly")
            else:
                print(f"⚠ Model {model_id} response: {response[:50]}...")
                
        except Exception as e:
            print(f"✗ Model {model_id} failed: {e}")

if __name__ == "__main__":
    print("Starting LangChain Assistant Tests...")
    
    # Run tests
    test_basic_functionality()
    test_error_handling()
    # test_different_models()  # Uncomment to test different models
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")