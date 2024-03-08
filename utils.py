from dotenv import load_dotenv

import sys


def load_env_file():
    """Load .env file. If not loaded, exit programme."""
    env_loaded = load_dotenv()

    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)
