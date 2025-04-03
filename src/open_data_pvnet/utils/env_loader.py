from dotenv import load_dotenv
from pathlib import Path

# Define the project base directory
PROJECT_BASE = Path(__file__).resolve().parent.parent.parent.parent


def load_environment_variables():
    """Load environment variables from a .env file."""
    env_path = PROJECT_BASE / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Environment variables loaded from {env_path}")
    else:
        raise FileNotFoundError(
            f"{env_path} file not found. Ensure you have a .env file in the project root."
        )
