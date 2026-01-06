#!/usr/bin/env python3
"""
Run script for the LangChain Assistant.
"""

import argparse
from src.main import main, chat_interactive


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LangChain Assistant")

    parser.add_argument(
        "--mode",
        choices=["demo", "interactive", "test"],
        default="demo",
        help="Run mode: demo (default), interactive, or test",
    )

    parser.add_argument(
        "--model", type=str, help="Specify a different Bedrock model ID"
    )

    parser.add_argument(
        "--temperature", type=float, help="Set temperature for model responses"
    )

    parser.add_argument(
        "--language", type=str, default="English", help="Default language for responses"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "demo":
        print("Running demo mode...")
        main()
    elif args.mode == "interactive":
        print("Starting interactive mode...")
        chat_interactive()
    elif args.mode == "test":
        print("Running tests...")
        # Import and run test script
        from scripts.test_assistant import test_basic_functionality

        test_basic_functionality()
