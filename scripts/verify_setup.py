#!/usr/bin/env python3
"""
Verification script for the LangChain Assistant project.
Checks all success criteria for Parts 1-6.
"""

import os
import sys
import subprocess

def check_part1():
    """Check Part 1: Project Setup"""
    print("="*60)
    print("Part 1: Project Setup Verification")
    print("="*60)
    
    checks = {
        "Project directory structure": os.path.exists("src") and os.path.exists("tests"),
        "requirements.txt exists": os.path.exists("requirements.txt"),
        ".gitignore exists": os.path.exists(".gitignore"),
        ".github/workflows directory exists": os.path.exists(".github/workflows"),
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            content = f.read()
            required_packages = ["langchain-aws", "langchain-core", "boto3", "python-dotenv", "pytest"]
            for package in required_packages:
                present = package in content
                status = "✓" if present else "✗"
                print(f"{status} {package} in requirements.txt")
    
    return all(checks.values())

def check_part2():
    """Check Part 2: Basic LangChain Application"""
    print("\n" + "="*60)
    print("Part 2: Basic LangChain Application Verification")
    print("="*60)
    
    checks = {
        "src/bedrock_client.py exists": os.path.exists("src/bedrock_client.py"),
        "src/chain.py exists": os.path.exists("src/chain.py"),
        "src/main.py exists": os.path.exists("src/main.py"),
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

def check_part3():
    """Check Part 3: Multiple Prompt Templates"""
    print("\n" + "="*60)
    print("Part 3: Multiple Prompt Templates Verification")
    print("="*60)
    
    checks = {
        "src/prompts.py exists": os.path.exists("src/prompts.py"),
        "PromptFactory class exists": False,  # Will check below
        "Multiple prompt templates": False,
    }
    
    if checks["src/prompts.py exists"]:
        with open("src/prompts.py") as f:
            content = f.read()
            checks["PromptFactory class exists"] = "class PromptFactory" in content
            checks["Multiple prompt templates"] = "SUPPORTED_TASKS" in content and "create_assistant_prompt" in content and "create_summarizer_prompt" in content
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

def check_part4():
    """Check Part 4: Output Parsing"""
    print("\n" + "="*60)
    print("Part 4: Output Parsing Verification")
    print("="*60)
    
    checks = {
        "StrOutputParser in chain.py": False,
    }
    
    if os.path.exists("src/chain.py"):
        with open("src/chain.py") as f:
            content = f.read()
            checks["StrOutputParser in chain.py"] = "StrOutputParser" in content and "|" in content
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

def check_part5():
    """Check Part 5: Basic Testing"""
    print("\n" + "="*60)
    print("Part 5: Basic Testing Verification")
    print("="*60)
    
    checks = {
        "tests/test_prompts.py exists": os.path.exists("tests/test_prompts.py"),
        "Tests import PromptFactory": False,
        "Tests check prompt formatting": False,
        "Tests check invalid prompts": False,
    }
    
    if checks["tests/test_prompts.py exists"]:
        with open("tests/test_prompts.py") as f:
            content = f.read()
            checks["Tests import PromptFactory"] = "PromptFactory" in content
            checks["Tests check prompt formatting"] = "format" in content
            checks["Tests check invalid prompts"] = "ValueError" in content or "invalid" in content.lower()
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

def check_part6():
    """Check Part 6: GitHub Actions CI/CD"""
    print("\n" + "="*60)
    print("Part 6: GitHub Actions CI/CD Verification")
    print("="*60)
    
    checks = {
        ".github/workflows/lint.yml exists": os.path.exists(".github/workflows/lint.yml"),
        ".github/workflows/test.yml exists": os.path.exists(".github/workflows/test.yml"),
        "lint.yml uses flake8": False,
        "test.yml uses pytest": False,
        "test.yml has matrix strategy": False,
    }
    
    if checks[".github/workflows/lint.yml exists"]:
        with open(".github/workflows/lint.yml") as f:
            content = f.read()
            checks["lint.yml uses flake8"] = "flake8" in content
    
    if checks[".github/workflows/test.yml exists"]:
        with open(".github/workflows/test.yml") as f:
            content = f.read()
            checks["test.yml uses pytest"] = "pytest" in content
            checks["test.yml has matrix strategy"] = "matrix" in content and "python-version" in content
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

def run_tests():
    """Run pytest to verify tests pass"""
    print("\n" + "="*60)
    print("Running Tests with pytest")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("✗ Tests failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def main():
    """Run all verification checks"""
    print("LangChain Assistant Project Verification")
    print("="*60)
    
    results = {
        "Part 1 - Project Setup": check_part1(),
        "Part 2 - Basic Application": check_part2(),
        "Part 3 - Multiple Prompts": check_part3(),
        "Part 4 - Output Parsing": check_part4(),
        "Part 5 - Basic Testing": check_part5(),
        "Part 6 - CI/CD": check_part6(),
        "Tests Pass": run_tests(),
    }
    
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_passed = True
    for part, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {part}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All verification checks passed!")
        print("Project meets all success criteria.")
    else:
        print("❌ Some verification checks failed.")
        print("Review the failed items above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
    