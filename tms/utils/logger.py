import json
import logging
import logging.config
import os
from pathlib import Path

from tms.utils.env import get_env


def setup_logging(
    path: str = "logging_config.json",
    level: int = logging.DEBUG,
    env_key: str = "LOG_CFG",
):
    """Setup logging configuration"""
    value = get_env(env_key, None)
    if value:
        path = value

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)
            # Add process ID to filename if using file handler
            if "file" in config.get("handlers", {}):
                process_id = os.getpid()
                config["handlers"]["file"]["filename"] = (
                    f"logs/tmsignatures_{process_id}.log"
                )
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)


def get_logger(name: str) -> logging.Logger:
    """Get logger with the given name"""
    return logging.getLogger(name)
