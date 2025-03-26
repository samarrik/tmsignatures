import concurrent.futures
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import gdown
import numpy as np
import pandas as pd
import requests
import torch
import yt_dlp
from torch.utils.data import Dataset
from tqdm import tqdm
from yt_dlp.utils import download_range_func

from tms.data.utils.video_manipulations import (
    compress_video,
    crop_talking_head,
    rescale_video,
    to_bytes,
    to_tensor,
    unify_processed_video_file,
)


class TMSDataset(Dataset):
    """
    Dataset class that stores, manages, and processes a dataset of talking head videos.
    """

    def __init__(self, path: Path):
        """
        Initializes the dataset object:
            - Sets up paths
            - Loads metadata
            - Loads status
            - Loads technical information
            - Loads data information
            - Browses dataset to load samples paths

        Args:
            path (Path): The path to the dataset.
        """
        # Setting up paths
        self.path = path
        self.data_path = path / "data"

        # Loading dataset metadata
        self.metadata = self._load_metadata()

        # Loading dataset status
        self.collected = self.metadata["status"]["collected"]
        self.preprocessed = self.metadata["status"]["preprocessed"]
        self.processed = self.metadata["status"]["processed"]
        self.postprocessed = self.metadata["status"]["postprocessed"]

        # Loading dataset technical information
        self.req_clip_length = self.metadata["tech_info"]["req_clip_length"]
        self.req_fps = self.metadata["tech_info"]["req_fps"]
        self.req_resolution = self.metadata["tech_info"]["req_resolution"]
        self.req_quality = self.metadata["tech_info"]["req_quality"]

        # Loading dataset data information
        self.transforms = [("Raw", "None")]
        if self.metadata["data_info"]["transforms"]:
            self.transforms += [
                (name, value)
                for name, values in self.metadata["data_info"]["transforms"].items()
                for value in values
            ]
        self.features = self.metadata["data_info"]["features"]
        self.sources = self.metadata["data_info"]["sources"]

        # Loading samples paths
        print("Loading samples paths...")
        self.samples_paths = list(self.data_path.rglob("*.mp4"))

    def _load_metadata(self):
        """
        Loads the metadata of the dataset.

        Returns:
            dict: The metadata of the dataset.
        """
        path = self.path / "metadata.json"
        json_data = {}
        with open(path, "r") as f:
            json_data = json.load(f)
        return json_data

    def _write_metadata(self):
        """
        Writes the metadata of the dataset.

        Only the status is being changed, nothing else can be updated
        """
        self.metadata["status"]["collected"] = self.collected
        self.metadata["status"]["preprocessed"] = self.preprocessed
        self.metadata["status"]["processed"] = self.processed
        self.metadata["status"]["postprocessed"] = self.postprocessed

        path = self.path / "metadata.json"
        with open(path, "w") as f:
            json.dump(
                self.metadata,
                f,
                indent=4,
            )

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples_paths)

    def __getitem__(self, idx):
        """
        Returns the path to the video at the given index. Doesn't apply transformations.

        Args:
            idx (int): The index of the video.

        Returns:
            Path: The path to the video at the given index.
        """
        return self.samples_paths[idx]

    def collect(self):
        """
        Collects the videos from the source.

        The videos are downloaded from the source and saved in the data directory.
        """

        def download_clip(
            video_id: str,
            output: str,
            start: int,
            end: int,
            max_retries: int = 5,
            retry_delay: int = 1,
        ):
            """Downloads a clip from a video.

            Args:
                video_id (str): The ID of the video.
                output (str): The path to save the clip.
                start (int): The start time of the clip.
                end (int): The end time of the clip.
                max_retries (int, optional): The maximum number of retries. Defaults to 5.
                retry_delay (int, optional): The delay between retries. Defaults to 1.

            Raises:
                RuntimeError: If the clip cannot be downloaded after the maximum number of retries.
            """
            max_retries = 5
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    ydl_opts = {
                        "outtmpl": str(output),
                        "download_ranges": download_range_func(None, [(start, end)]),
                        "force_keyframes_at_cuts": True,
                        "format": "best[height<=480][ext=mp4]/best[ext=mp4]/best",  # More flexible format selection
                        "quiet": True,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download(f"https://www.youtube.com/watch?v={video_id}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to download video {video_id} after {max_retries} attempts: {e}"
                        )
                    time.sleep(retry_delay)

        print("Collecting videos...")
        # Create data subdirectories
        train_dir = self.data_path / "train"
        test_dir = self.data_path / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Process train and test sets
        try:
            for split in ["train", "test"]:
                split_data = self.sources[split]
                split_dir = train_dir if split == "train" else test_dir

                # Process each identity
                for identity_id, identity_data in tqdm(
                    split_data.items(),
                    desc=f"Downloading {split} videos of identities",
                ):
                    identity_dir = split_dir / identity_id
                    identity_dir.mkdir(exist_ok=True)

                    # Process each video
                    for video_id, video_data in identity_data.items():
                        video_dir = identity_dir / video_id
                        video_dir.mkdir(exist_ok=True)

                        clips = video_data["clips"]

                        # Download and process video clips
                        for clip_idx, clip_data in clips.items():
                            clip_path = video_dir / f"{clip_idx}.mp4"
                            start = sum(
                                int(x) * 60**i
                                for i, x in enumerate(
                                    reversed(clip_data["start"].split(":"))
                                )
                            )  # HH:MM:SS converted to seconds
                            end = sum(
                                int(x) * 60**i
                                for i, x in enumerate(
                                    reversed(clip_data["end"].split(":"))
                                )
                            )  # HH:MM:SS converted to seconds

                            download_clip(video_id, clip_path, start, end)
        except Exception as e:
            print(f"Error collecting videos: {e}")
            return

        print("Videos collected successfully.")
        self.collected = True
        self._write_metadata()

    def preprocess(self):
        """
        Cuts out the talking head from the video, controls and adjusts desired fps and resolution.
        """
        print("Preprocessing videos...")
        cpu_cores = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
        if cpu_cores == 1 and os.cpu_count() > 1:  # If SLURM_CPUS_PER_TASK is not set
            cpu_cores = min(os.cpu_count(), 16)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            # Start processing all videos in parallel
            futures = {
                executor.submit(
                    crop_talking_head,
                    sample_path,
                    fps=self.req_fps,
                    resolution=self.req_resolution,
                ): sample_path
                for sample_path in self.samples_paths
            }

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Preprocessing videos",
            ):
                sample_path = futures[future]
                try:
                    future.result()  # Get result to catch any exceptions
                except Exception as e:
                    print(f"Error preprocessing {sample_path}: {e}")
                    return

        print("Videos preprocessed successfully.")
        self.preprocessed = True
        self._write_metadata()

    def process(self):
        """
        Process video samples with all given transformations.
        """
        print("Processing videos...")
        cpu_cores = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
        if cpu_cores == 1 and os.cpu_count() > 1:  # If SLURM_CPUS_PER_TASK is not set
            cpu_cores = min(os.cpu_count(), 16)

        gpus = int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
        if gpus == 0:
            raise ValueError(
                "No GPUs available - extremly long computation time expected, aborting..."
            )

        # Choose only samples that have not been processed
        unprocessed_items = []
        for transform_idx, (transform_name, transform_value) in enumerate(
            self.transforms
        ):
            transform_dir = (
                self.path / "processed" / transform_name / str(transform_value)
            )

            for sample_path in self.samples_paths:
                inner_structure = Path(sample_path).relative_to(self.data_path).parent
                output_path = (
                    transform_dir / inner_structure / f"{Path(sample_path).stem}.csv"
                )

                if not output_path.exists():
                    unprocessed_items.append((transform_idx, sample_path, output_path))

        if not unprocessed_items:
            print("All samples have been processed already.")
            self.processed = True
            self._write_metadata()
            return
        else:
            print(
                f"Found {len(unprocessed_items)} items to process out of {len(self.samples_paths) * len(self.transforms)} total"
            )

        # Distribute items by gpu
        items_by_gpu = {}
        for i, item in enumerate(unprocessed_items):
            gpu_id = i % gpus
            if gpu_id not in items_by_gpu:
                items_by_gpu[gpu_id] = []
            items_by_gpu[gpu_id].append(item)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=gpus  # one process per GPU
        ) as executor:
            # Submit jobs for each GPU
            futures = []
            for gpu_id, items in items_by_gpu.items():
                future = executor.submit(
                    TMSDataset._process_subset,
                    gpu_id,
                    items,
                    self.transforms,
                    cpu_cores,
                )
                futures.append(future)

            # Wait for all processes to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Processing failed with error: {e}")
                    return

        print("Videos processed successfully.")
        self.processed = True
        self._write_metadata()

    @staticmethod
    def _process_subset(
        gpu_id: int,
        items_to_process: list,
        transforms: list,
        num_workers: int,
    ) -> None:
        """
        Process a subset of video samples with all given transformations.

        Args:
            gpu_id (int): The ID of the GPU to use.
            items_to_process (list): The list of items to process.
            transforms (list): The list of transformations to apply.
            num_workers (int): The number of workers to use.
        """
        # Set up GPU
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        # Initialize detector
        from tms.data.utils.detector import Detector  #! avoiding pickle issues

        detector = Detector()

        # Process assigned work items
        for transform_idx, sample_path, output_path in tqdm(
            items_to_process,
            desc=f"Processing on GPU {gpu_id}",
            position=gpu_id,
        ):
            transform_name, transform_value = transforms[transform_idx]

            try:
                # Create output directory
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Compose transformation
                bytes = to_bytes(sample_path)
                if transform_name == "Compress":
                    transformed_bytes = compress_video(bytes, transform_value)
                elif transform_name == "Rescale":
                    transformed_bytes = rescale_video(bytes, transform_value)
                elif transform_name == "Raw":
                    transformed_bytes = bytes
                else:
                    raise ValueError(f"Unknown transformation: {transform_name}")
                processed_tensor = to_tensor(transformed_bytes)

                # Extract features
                features = detector.detect(
                    processed_tensor,
                    batch_size=100,
                    num_workers=num_workers,
                    face_detection_threshold=0.1,
                    progress_bar=False,  # Disable nested progress bars
                )

                # Save features
                features.to_csv(output_path, index=False)

            except Exception as e:
                print(
                    f"Error processing {sample_path} with transform {transform_name}({transform_value}): {e}"
                )
                continue

    def postprocess(self):
        """
        Postprocesses the videos by computing the correlations for all the different length
        coefficients for train and test subsets.
        """
        print("Postprocessing videos...")
        try:
            # Process all length coefficients in parallel using a process pool for train and test subsets
            length_coeffs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.01]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Compute correlations for all the different length coefficients for train and test
                train_args = [
                    (self, length_coeff, "train") for length_coeff in length_coeffs
                ]
                test_args = [
                    (self, length_coeff, "test") for length_coeff in length_coeffs
                ]

                futures = {
                    executor.submit(TMSDataset._compute_correlations, *args): args
                    for args in train_args + test_args
                }

                for future in concurrent.futures.as_completed(futures):
                    args = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing {args}: {e}")
                        return

            print("Videos postprocessed successfully.")
            self.postprocessed = True
            self._write_metadata()
        except Exception as e:
            print(f"Error during postprocessing: {e}")

    @staticmethod
    def _compute_correlations(dataset, length_coeff: float, subset: str):
        """
        Computes the correlations for all the different length coefficients for train and test subsets.

        Args:
            dataset: The dataset to process
            length_coeff: The length coefficient to use
            subset: The subset to process ("train" or "test")
        """
        processed_path = dataset.path / "processed"
        correlations_path = dataset.path / "correlations"

        # Define output path early to check if it exists
        output_path = correlations_path / f"{length_coeff}_{subset}.csv"

        # Check if the file already exists
        if output_path.exists():
            print(
                f"Correlations file already exists for {subset} subset with length coefficient {length_coeff}. Skipping..."
            )
            return

        os.makedirs(correlations_path, exist_ok=True)

        subset_processed_files_paths = list(
            str(p) for p in processed_path.rglob("*.csv") if subset in str(p)
        )

        if len(subset_processed_files_paths) == 0:
            print(f"No processed files found for {subset} subset")
            return

        # Get the features, clip length and fps from the dataset
        features = dataset.features
        clip_length = dataset.req_clip_length
        fps = dataset.req_fps

        df_corr = pd.DataFrame()

        subclip_length = clip_length * length_coeff

        for path in tqdm(
            subset_processed_files_paths,
            desc=f"Postprocessing {subset} subset of {dataset.name}",
        ):
            try:
                # Extract metadata from path
                path_parts = os.path.normpath(path).split(os.sep)
                if len(path_parts) < 6:
                    print(f"Skipping file with invalid path structure: {path}")
                    continue

                identity = path_parts[-3]
                compress_value = int(path_parts[-5]) if "Compress" in path else 0
                rescale_value = float(path_parts[-5]) if "Rescale" in path else 0

                df_file = unify_processed_video_file(path, features, clip_length, fps)

                # Compute clip boundaries
                fcv = int(len(df_file))
                fcc = int(min(fps * subclip_length, fcv))

                if fcc <= 0:
                    print(f"Skipping file with invalid clip length: {path}")
                    continue

                # Sample and process subclip
                clip_start = 0 if fcv == fcc else int(np.random.randint(0, fcv - fcc))

                df_subclip = df_file.iloc[clip_start : clip_start + fcc].copy()

                # Compute the correlation matrix
                corr_matrix = df_subclip.corr().fillna(
                    0
                )  # Fill NaNs in the correlation matrix, may appear because the values don't change

                # Flatten correlation matrix into unique pairs
                corr_pairs = {
                    f"{min(col1, col2)}*{max(col1, col2)}": corr_matrix.loc[col1, col2]
                    for col1 in corr_matrix.columns
                    for col2 in corr_matrix.columns
                    if col1 < col2  # Only take unique pairs
                }

                # Add identity, compress, and rescale info
                corr_pairs["identity"] = identity
                corr_pairs["compress"] = compress_value
                corr_pairs["rescale"] = rescale_value

                # Add to df_corr DataFrame
                df_corr = pd.concat([df_corr, pd.DataFrame([corr_pairs])])

            except Exception as e:
                print(f"Error processing file {path}: {str(e)}")
                continue

        # Save the computed df_corr for this clip configuration
        # Save without index to avoid 'Unnamed: 0' column in the output
        df_corr.to_csv(output_path, index=False)


class FacialFeaturesDataset(Dataset):
    """
    Dataset of only the facial features of the videos. Primarily used for training and testing NNs.
    """

    def __init__(
        self,
        general_dataset: TMSDataset,
    ):
        self.name = general_dataset.name
        self.samples = general_dataset.samples_paths
        self.identity_to_label = {
            id_: i for i, id_ in enumerate(sorted(set(s[1] for s in self.samples)))
        }
        self.features = general_dataset.features
        self.clip_length = general_dataset.req_clip_length
        self.fps = general_dataset.req_fps

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Returns the features and the label for the sample at the given index.
        """
        path, identity = self.samples[idx]
        features = unify_processed_video_file(
            path, self.features, self.clip_length, self.fps
        ).values.astype(np.float32)
        return features, self.identity_to_label[identity]


def initialize_datasets(data_dir_path: Path = Path("data/")) -> Dict[str, TMSDataset]:
    """
    Initialize datasets from predownloaded data.

    Args:
        data_dir_path: Path to the data directory

    Returns:
        Dictionary mapping dataset names to dataset objects
    """
    print("Initializing datasets from predownloaded data...")
    if not data_dir_path.exists() or not any(data_dir_path.iterdir()):
        download_datasets()

    datasets = [TMSDataset(d) for d in data_dir_path.iterdir() if d.is_dir()]
    print(f"Datasets initialized successfully. Found {len(datasets)} datasets.")
    return {d.name: d for d in datasets}


def download_datasets(
    url: Optional[str] = None, output: Path = Path("archive.zip")
) -> Dict[str, TMSDataset]:
    """
    Download datasets from the given URL or Google Drive ID.

    Args:
        url: URL or Google Drive ID to download from
        output: Path to save the downloaded file

    Returns:
        Dictionary mapping dataset names to dataset objects
    """

    print("Downloading datasets...")
    data_dir = Path("data/")

    # Preserving old data
    if data_dir.exists():
        print(
            f"Warning: {data_dir} already exists. Renaming to {data_dir.with_suffix('.old')}"
        )
        data_dir.rename(data_dir.with_suffix(".old"))

    while url is None:
        url = input("Enter the datasets URL or Google Drive ID: ")

        # Extract file ID from Google Drive URL if needed
        if "drive.google.com" in url and "file/d/" in url:
            file_id = url.split("file/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?id={file_id}"
            break
        elif "drive.google.com" in url and "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
            url = f"https://drive.google.com/uc?id={file_id}"
            break

        # Validate the url for non-Google Drive URLs
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                break
            else:
                print("Invalid URL or network error. Please try again.")
                url = None
        except requests.RequestException:
            print("Invalid URL or network error. Please try again.")
            url = None

    # Important: Convert Path to string for gdown
    output_str = str(output)

    # Use gdown with proper parameters
    gdown.download(url, output_str, quiet=False, fuzzy=True)

    # Unzip the file
    with zipfile.ZipFile(output_str, "r") as zip_ref:
        zip_ref.extractall(output.parent)

    # Remove the zip file
    output.unlink()
    print("Datasets downloaded successfully.")

    datasets = initialize_datasets()

    return datasets
