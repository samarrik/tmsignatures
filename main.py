from tms.utils.env import setup_environment
from tms.utils.logger import get_logger, setup_logging

setup_environment()
setup_logging()

from tms.experiments import perform_experiments
from tms.extraction import perform_extraction
from tms.postprocessing import perform_postprocessing
from tms.utils.data import initialize_datasets

logger = get_logger(__name__)


def main():
    logger.info("Initializing datasets...")
    datasets = initialize_datasets()

    # logger.info("Extracting features...")
    # perform_extraction(datasets)

    # logger.info("Postprocessing features...")
    # perform_postprocessing(datasets)

    # logger.info("Running experiments...")
    # perform_experiments(datasets)

    logger.info("Done.")


if __name__ == "__main__":
    main()
