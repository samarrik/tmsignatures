from tms.utils.env import setup_environment
from tms.utils.logger import get_logger, setup_logging

setup_environment()
setup_logging()

logger = get_logger(__name__)


def main():
    logger.info("Starting TMSignatures")


if __name__ == "__main__":
    main()
