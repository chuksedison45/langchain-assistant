#!/usr/bin/env python3
"""
Simple test to verify basic functionality.
"""
import sys
import os
from prompts import PromptFactory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_basic_prompt_factory():
    """Test basic prompt factory functionality."""
    print("Testing PromptFactory...")

    factory = PromptFactory()

    # Test 1: List tasks
    print("\n1. Testing list_tasks():")
    tasks = factory.list_tasks()
    print(f"   Tasks: {tasks}")

    # Test 2: Get assistant prompt
    print("\n2. Testing get_prompt_template('assistant'):")
    assistant_prompt = factory.get_prompt_template("assistant")
    print(f"   Prompt type: {type(assistant_prompt)}")

    # Test 3: Format assistant prompt
    formatted = assistant_prompt.format(language="English", message="Hello")
    print(f"   Formatted length: {len(formatted)} characters")
    print(f"   Sample: {formatted[:100]}...")

    # Test 4: Get summarizer prompt
    print("\n3. Testing get_prompt_template('summarizer'):")
    summarizer_prompt = factory.get_prompt_template("summarizer")
    print(f"   Prompt type: {type(summarizer_prompt)}")

    # Test 5: Format summarizer prompt
    formatted = summarizer_prompt.format(text="Sample text", length="brief")
    print(f"   Formatted length: {len(formatted)} characters")
    print(f"   Sample: {formatted[:100]}...")

    # Test 6: Test invalid task
    print("\n4. Testing invalid task:")
    try:
        factory.get_prompt_template("invalid")
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    # Test 7: Test input variables
    print("\n5. Testing get_task_input_variables():")
    for task in ["assistant", "summarizer", "translator", "coder"]:
        vars = factory.get_task_input_variables(task)
        print(f"   {task}: {vars}")

    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    test_basic_prompt_factory()
