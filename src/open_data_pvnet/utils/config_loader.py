import logging
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """

    if not config_path:
        raise ValueError("Missing config path")

    # Load and return the configuration
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
