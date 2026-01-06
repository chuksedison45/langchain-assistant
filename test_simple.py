#!/usr/bin/env python3
"""
Simple test script for quick verification.
"""

import sys
sys.path.append('src')

from src.main import LangChainAssistant

def quick_test():
    """Quick test with two different languages."""
    
    print("🔧 Quick Test - LangChain Assistant")
    print("-" * 40)
    
    # Create assistant (less verbose for quick test)
    assistant = LangChainAssistant(verbose=False)
    
    # Test 1: English
    print("\n1. Testing English response...")
    response_en = assistant.chat(
        "Explain quantum computing in simple terms.",
        "English"
    )
    print(f"   English response received: {len(response_en)} chars")
    
    # Test 2: Spanish
    print("\n2. Testing Spanish response...")
    response_es = assistant.chat(
        "¿Qué es la computación cuántica?",
        "Spanish"
    )
    print(f"   Spanish response received: {len(response_es)} chars")
    
    # Test 3: French
    print("\n3. Testing French response...")
    response_fr = assistant.chat(
        "Parle-moi de l'apprentissage automatique.",
        "French"
    )
    print(f"   French response received: {len(response_fr)} chars")
    
    print("\n" + "=" * 40)
    print("✅ Quick test completed successfully!")
    print(f"   Languages tested: English, Spanish, French")
    
    # Show a snippet of each response
    print("\nResponse snippets:")
    print(f"  EN: {response_en[:100]}...")
    print(f"  ES: {response_es[:100]}...")
    print(f"  FR: {response_fr[:100]}...")

if __name__ == "__main__":
    quick_test()
    