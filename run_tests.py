#!/usr/bin/env python3
"""
Test runner script for the LangChain Assistant project.
Runs all tests without requiring AWS credentials.
"""

import subprocess
import sys
import os


def run_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Running LangChain Assistant Tests")
    print("=" * 70)

    # Run pytest with verbose output
    test_dir = os.path.join(os.path.dirname(__file__), "tests")

    # Command to run pytest
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_dir,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--disable-warnings",  # Disable warnings for cleaner output
    ]

    print(f"\nRunning command: {' '.join(cmd)}")
    print("-" * 70)

    # Run the tests
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print output
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print("-" * 70)

    # Return exit code
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {result.returncode}")

    return result.returncode


def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"\nRunning specific test: {test_file}")
    print("-" * 70)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--disable-warnings",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tests for LangChain Assistant")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run tests with coverage report"
    )

    args = parser.parse_args()

    if args.file:
        # Run specific test file
        exit_code = run_specific_test(args.file)
    else:
        # Run all tests
        if args.coverage:
            print("Running tests with coverage...")
            # Add coverage command
            coverage_cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=term-missing",
                "-v",
            ]
            result = subprocess.run(coverage_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            exit_code = result.returncode
        else:
            exit_code = run_tests()

    sys.exit(exit_code)
