import sys
import boto3
from dotenv import load_dotenv
import os


def test_environment():
    """Test that environment and dependencies are properly set up."""
    print("Python version:", sys.version)

    # Test imports
    try:
        print("✓ All required packages imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Test environment variables
    load_dotenv()
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_profile:
        print(f"✓ AWS Profile configured: {aws_profile}")
    else:
        print("⚠ AWS_PROFILE not set in .env")

    # Test boto3 session
    try:
        session = boto3.Session(profile_name=aws_profile)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid for: {identity['Arn']}")
    except Exception as e:
        print(f"✗ AWS credentials error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
