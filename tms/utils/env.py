import os
from pathlib import Path

from dotenv import load_dotenv


def setup_environment(env_path: str = None) -> None:
    """Setup environment variables from .env file"""
    # Default locations to search for .env file
    env_locations = [
        env_path,
        ".env",
    ]

    # Find first existing .env file
    for env_file in env_locations:
        if env_file and Path(env_file).is_file():
            load_dotenv(env_file)
            return

    # If no .env file found, create default one
    create_default_env()


def create_default_env(env_path: str = ".env") -> None:
    """Create default .env file if it doesn't exist"""
    if Path(env_path).is_file():
        return

    default_env = {
        # Environment
        "ENV": "development",
        "LOG_LEVEL": "INFO",
        # Paths
        "DATA_DIR": "data",
        "MODELS_DIR": "models",
        "FIGURES_DIR": str(Path("reports") / "figures"),
        "PICTURES_DIR": str(Path("reports") / "media" / "pictures"),
        "VIDEOS_DIR": str(Path("reports") / "media" / "videos"),
        "LOGS_DIR": "logs",
        "TEMP_DIR": "temp",
        "LOG_CFG": "logging_config.json",
        "DATASETS_CFG": "datasets_config.json",
    }

    with open(env_path, "w") as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with default fallback"""
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    try:
        return float(os.getenv(key, default))
    except ValueError:
        return default


def get_env_list(key: str, default: list = None, separator: str = ",") -> list:
    """Get list environment variable"""
    value = os.getenv(key)
    if value is None:
        return default if default is not None else []
    return [item.strip() for item in value.split(separator)]
