import datetime
import logging
import math
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from tms.detector import Detector
from tms.utils.data import TMSDataset

logger = logging.getLogger(__name__)


def extract_features(
    datasets: List[TMSDataset],
    subdataset_id: int,
    subdatasets_cnt: int,
    num_workers: int = 0,
) -> None:
    """Extract features from video datasets.

    This function processes a subset of samples from multiple datasets,
    extracting features using the Detector class. The subset is determined
    by subdataset_id and subdatasets_cnt parameters.

    Args:
        datasets: List of datasets to process.
        subdataset_id: ID of the current subdataset (0 to subdatasets_cnt-1).
        subdatasets_cnt: Total number of subdatasets to split the work into.
        num_workers: Number of worker processes for feature extraction.
            Default: 0 (single process).

    Raises:
        ValueError: If datasets list is empty.
    """
    if not datasets:
        raise ValueError("Datasets list is empty")

    # Calculate total samples including all transforms
    total_samples = 0
    for dataset in datasets:
        dataset_size = len(dataset.samples)
        num_transforms = len(dataset.transforms) + 1 if dataset.transforms else 1
        total_samples += dataset_size * num_transforms

    # Calculate subdataset boundaries
    samples_per_subdataset = math.ceil(total_samples / subdatasets_cnt)
    start_idx = subdataset_id * samples_per_subdataset
    end_idx = min(start_idx + samples_per_subdataset, total_samples)

    # Collect samples for this subdataset
    subdataset_samples = _collect_subdataset_samples(datasets, start_idx, end_idx)

    # Initialize detector
    detector = Detector()

    # Process samples
    start_time = datetime.datetime.now()
    logger.info(
        f"Starting feature extraction at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(subdataset {subdataset_id}/{subdatasets_cnt})"
    )

    for dataset, sample_idx, transform_idx in tqdm(
        subdataset_samples,
        desc=f"Extracting features (subdataset {subdataset_id}/{subdatasets_cnt})",
        unit="sample",
    ):
        dataset.transform_in_use_idx = transform_idx
        _process_sample(dataset, sample_idx, detector, num_workers)

    # Log completion
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(
        f"Completed feature extraction at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(duration: {duration:.2f}s)"
    )


def _collect_subdataset_samples(
    datasets: List[TMSDataset],
    start_idx: int,
    end_idx: int,
) -> List[Tuple[TMSDataset, int, int]]:
    """Collect samples that belong to the current subdataset.

    Args:
        datasets: List of datasets to process.
        start_idx: Global start index for this subdataset.
        end_idx: Global end index for this subdataset.

    Returns:
        List of (dataset, sample_index, transform_index) tuples to process.
    """
    samples = []
    current_idx = 0

    for dataset in datasets:
        dataset_size = len(dataset.samples)
        num_transforms = len(dataset.transforms) + 1 if dataset.transforms else 1
        total_dataset_size = dataset_size * num_transforms

        # Skip if this dataset ends before our range
        if current_idx + total_dataset_size <= start_idx:
            current_idx += total_dataset_size
            continue

        # Stop if we've passed our range
        if current_idx >= end_idx:
            break

        # Calculate which samples from this dataset to include
        dataset_start = max(0, start_idx - current_idx)
        dataset_end = min(total_dataset_size, end_idx - current_idx)

        # Convert global indices to (sample_idx, transform_idx) pairs
        for global_idx in range(dataset_start, dataset_end):
            sample_idx = global_idx % dataset_size
            transform_idx = (global_idx // dataset_size) - 1
            samples.append((dataset, sample_idx, transform_idx))

        current_idx += total_dataset_size

    return samples


def _process_sample(
    dataset: TMSDataset,
    sample_idx: int,
    detector: Detector,
    num_workers: int,
) -> None:
    """Process a single sample from the dataset.

    Args:
        dataset: Dataset containing the sample.
        sample_idx: Index of the sample in the dataset.
        detector: Initialized detector instance.
        num_workers: Number of worker processes to use.
    """
    # Get sample path and transform info
    sample_path = dataset.samples[sample_idx]
    transform_info = _get_transform_info(dataset)

    logger.debug(f"Processing sample: {sample_path} with transform: {transform_info}")

    # Setup output path
    output_path = _get_output_path(dataset, sample_path, transform_info)
    if output_path.exists():
        logger.debug(f"Skipping existing output: {output_path}")
        return

    # Extract features
    try:
        sample = dataset[sample_idx]
        features = detector.detect(
            sample,
            batch_size=100,
            num_workers=num_workers,
            face_detection_threshold=0.1,
            progress_bar=True,
        )

        # Save features
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)

    except Exception as e:
        logger.error(f"Failed to process {sample_path}: {str(e)}")


def _get_transform_info(dataset: TMSDataset) -> Tuple[str, str]:
    """Get current transform information.

    Args:
        dataset: Dataset to get transform info from.

    Returns:
        Tuple of (transform_name, transform_value).
    """
    if dataset.transform_in_use_idx == -1:
        return ("Raw", "None")

    transform_name, transform_value = dataset.transforms[dataset.transform_in_use_idx]
    return (transform_name, str(transform_value))


def _get_output_path(
    dataset: TMSDataset,
    sample_path: Path,
    transform_info: Tuple[str, str],
) -> Path:
    """Construct output path for extracted features.

    Args:
        dataset: Dataset containing the sample.
        sample_path: Path to the input sample.
        transform_info: Tuple of (transform_name, transform_value).

    Returns:
        Path where extracted features should be saved.
    """
    transform_name, transform_value = transform_info

    # Get relative path structure
    rel_path = Path(sample_path).relative_to(dataset.dataset_data_dir)

    # Construct output path
    output_path = (
        dataset.dataset_extracted_dir
        / transform_name
        / transform_value
        / rel_path.parent
        / f"{rel_path.stem}.csv"
    )

    return output_path
