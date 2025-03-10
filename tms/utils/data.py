import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import time

import ffmpeg
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tms.utils.env import get_env
from tms.utils.facecrop import crop_face_from_video

logger = logging.getLogger(__name__)


class ToBytes:
    """Transform a video file path to raw RGB bytes.

    This transformer reads a video file and converts it to raw RGB24 bytes
    using FFmpeg, maintaining the original dimensions and ensuring consistent
    frame count through proper frame extraction.
    """

    def __init__(self, target_fps: int = 25):
        """Initialize the transformer.

        Args:
            target_fps: Target framerate to extract frames at. Default: 25.
        """
        self.target_fps = target_fps

    def __call__(self, sample: Union[str, Path]) -> Tuple[bytes, int, int]:
        """Convert video to raw RGB bytes.

        Args:
            sample: Path to the video file.

        Returns:
            Tuple containing:
                - Raw video bytes in RGB24 format
                - Video width
                - Video height

        Raises:
            RuntimeError: If FFmpeg conversion fails.
        """
        try:
            # Get video information
            probe = ffmpeg.probe(str(sample))
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            width = int(video_info["width"])
            height = int(video_info["height"])

            # Calculate actual duration and expected frame count
            duration = float(probe["format"]["duration"])
            expected_frames = int(duration * self.target_fps)

            # Get raw bytes with explicit frame extraction
            out, _ = (
                ffmpeg.input(str(sample))
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    r=self.target_fps,  # Force constant frame rate
                    vsync="cfr",  # Force constant frame rate
                    vf=f"fps={self.target_fps}",  # Extract at exact intervals
                )
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Verify frame count
            actual_frames = len(out) // (width * height * 3)
            if actual_frames != expected_frames:
                logger.warning(
                    f"Frame count mismatch in {sample}: "
                    f"expected {expected_frames}, got {actual_frames}"
                )

            return out, width, height

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Converting to bytes failed for {sample}: {error_msg}")


class Rescale:
    """Transform to rescale video dimensions by a coefficient.

    This transformer takes raw video bytes and rescales the dimensions
    while maintaining the aspect ratio.
    """

    def __init__(self, scale_coeff: float) -> None:
        """Initialize the rescaler.

        Args:
            scale_coeff: Scaling coefficient (0 < scale_coeff <= 1).
        """
        if not 0 < scale_coeff <= 1:
            raise ValueError("scale_coeff must be between 0 and 1")
        self.scale_coeff = scale_coeff

    def __call__(self, video_bytes: Tuple[bytes, int, int]) -> Tuple[bytes, int, int]:
        """Rescale video dimensions.

        Args:
            video_bytes: Tuple of (raw video bytes, width, height).

        Returns:
            Tuple containing:
                - Rescaled video bytes
                - New width
                - New height

        Raises:
            RuntimeError: If FFmpeg scaling fails.
        """
        try:
            video_data, width, height = video_bytes

            # Calculate new dimensions (ensure they're even)
            new_width = (int(width * self.scale_coeff) // 2) * 2
            new_height = (int(height * self.scale_coeff) // 2) * 2

            # Apply transformation and get bytes directly with ultrafast preset
            out, _ = (
                ffmpeg.input(
                    "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}"
                )
                .output(
                    "pipe:",
                    vf=f"scale={new_width}:{new_height}",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    preset="ultrafast",
                    threads="auto",
                )
                .run(input=video_data, capture_stdout=True, capture_stderr=True)
            )

            return out, new_width, new_height

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Scaling failed: {error_msg}")


class Compress:
    """Transform to compress video using H.264 encoding.

    This transformer compresses video using H.264 with specified CRF value,
    maintaining the input dimensions.
    """

    def __init__(self, crf_value: int) -> None:
        """Initialize the compressor.

        Args:
            crf_value: Constant Rate Factor value (0-51, lower means better quality).
        """
        if not 0 <= crf_value <= 51:
            raise ValueError("crf_value must be between 0 and 51")
        self.crf_value = crf_value

    def __call__(self, video_bytes: Tuple[bytes, int, int]) -> Tuple[bytes, int, int]:
        """Compress video using H.264.

        Args:
            video_bytes: Tuple of (raw video bytes, width, height).

        Returns:
            Tuple containing:
                - Compressed video bytes
                - Width (unchanged)
                - Height (unchanged)

        Raises:
            RuntimeError: If FFmpeg compression fails.
        """
        try:
            video_data, width, height = video_bytes

            # Direct pipe-to-pipe compression using more efficient settings
            out, _ = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{width}x{height}",
                    framerate=25,
                )
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    vcodec="libx264",
                    crf=self.crf_value,
                    preset="ultrafast",
                    movflags="+faststart",  # Optimize for streaming
                    g=25,  # Keyframe interval
                    threads="auto",
                )
                .run(input=video_data, capture_stdout=True, capture_stderr=True)
            )

            return out, width, height

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Compression failed: {error_msg}")


class ToTensor:
    """Transform to convert raw video bytes to PyTorch tensor.

    This transformer converts raw RGB24 video bytes to a 4D PyTorch tensor
    in NCHW format (frames, channels, height, width).
    """

    def __call__(self, video_bytes: Tuple[bytes, int, int]) -> torch.Tensor:
        """Convert video bytes to PyTorch tensor.

        Args:
            video_bytes: Tuple of (raw video bytes, width, height).

        Returns:
            4D PyTorch tensor in NCHW format (frames, channels, height, width).
        """
        video_data, width, height = video_bytes

        # More efficient tensor creation and reshape
        video = torch.frombuffer(video_data, dtype=torch.uint8).contiguous()
        video = video.view(-1, height, width, 3)  # Using view instead of reshape
        video = video.permute(
            0, 3, 1, 2
        ).contiguous()  # Make memory contiguous after permute

        return video


class TMSDataset(Dataset):
    """Dataset class for TMS video processing.

    This dataset handles video data with configurable transformations
    and features as specified in the datasets_config.json.
    """

    def __init__(
        self,
        dataset_name: str,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialize the TMS dataset.

        Args:
            dataset_name: Name of the dataset as specified in config.
            config_path: Optional path to dataset config file.
                        If None, uses DATASETS_CFG from environment.

        Raises:
            ValueError: If dataset_name is not found in config.
            FileNotFoundError: If config file or data directory not found.
        """
        self.dataset_name = dataset_name

        # Load configuration
        self.config = self._load_config(config_path)
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")

        # Set dataset-specific configuration
        self.dataset_config = self.config["datasets"][dataset_name]
        self.clip_length = self.dataset_config.get("clip_length", 30)
        self.fps = self.dataset_config.get("fps", 25)
        self.features = self.dataset_config.get("features", [])

        # Process transforms configuration
        transforms_config = self.dataset_config.get("transforms", {})
        self.transforms = [
            (name, value)
            for name, values in transforms_config.items()
            for value in values
        ]
        self.transform_in_use_idx = -1  # -1 means no transform

        # Setup paths
        data_dir = Path(get_env("DATA_DIR", "data"))
        self.dataset_dir = data_dir / dataset_name
        self.dataset_data_dir = self.dataset_dir / "data"
        self.dataset_extracted_dir = self.dataset_dir / "extracted"
        self.dataset_correlations_dir = self.dataset_dir / "correlations"

        # Ensure required directories exist
        self.dataset_data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_extracted_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_correlations_dir.mkdir(parents=True, exist_ok=True)

        # Check if dataset needs preprocessing
        if not self._is_dataset_processed():
            logger.info(
                f"Dataset {dataset_name} not found or incomplete. Starting preprocessing..."
            )
            self._preprocess_dataset()
            logger.info(f"Dataset {dataset_name} preprocessing completed.")

        # Load data
        self.samples = self._load_samples()
        logger.info(
            f"Initialized {dataset_name} dataset with {len(self.samples)} samples "
            f"and {len(self.transforms)} available transforms"
        )

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset by downloading and cropping videos."""
        # Create necessary directories
        train_dir = self.dataset_data_dir / "train"
        test_dir = self.dataset_data_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        sources_path = self.dataset_dir / "metadata" / "sources.json"
        if not sources_path.exists():
            raise FileNotFoundError(f"Sources file not found: {sources_path}")

        with open(sources_path) as f:
            sources = json.load(f)

        # Process train and test sets
        for split in ["train", "test"]:
            split_data = sources[split]
            split_dir = train_dir if split == "train" else test_dir

            # Process each identity
            for identity_id, identity_data in tqdm(
                split_data.items(), desc=f"Processing {split} identities"
            ):
                identity_dir = split_dir / identity_id
                identity_dir.mkdir(exist_ok=True)

                # Process each video
                for video_id, video_data in tqdm(
                    identity_data.items(), desc=f"Processing videos for {identity_id}"
                ):
                    youtube_id = video_data["id"]
                    video_dir = identity_dir / youtube_id
                    video_dir.mkdir(exist_ok=True)

                    clips = video_data["clips"]

                    # Download and process video clips
                    self._process_video_clips(
                        youtube_id=youtube_id,
                        clips=clips,
                        output_dir=video_dir,
                    )

    def _process_video_clips(
        self,
        youtube_id: str,
        clips: Dict[str, Dict[str, str]],
        output_dir: Path,
    ) -> None:
        """Process video clips from YouTube video.

        Args:
            youtube_id: YouTube video ID
            clips: Dictionary of clip information
            output_dir: Directory to save processed clips
        """
        # First check if all clips already exist
        all_clips_exist = True
        missing_clips = []
        for clip_id in clips.keys():
            clip_path = output_dir / f"{clip_id}.mp4"
            if not clip_path.exists():
                all_clips_exist = False
                missing_clips.append(clip_id)

        if all_clips_exist:
            return

        try:
            # Download video in 480p using yt-dlp with retries
            temp_video = output_dir / "temp_video.mp4"
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    cmd = [
                        "yt-dlp",
                        f"https://www.youtube.com/watch?v={youtube_id}",
                        "-f", "best[height<=480][ext=mp4]",
                        "-o", str(temp_video),
                        "--no-playlist",
                        "--quiet"
                    ]
                    subprocess.run(cmd, check=True)
                    break
                except subprocess.CalledProcessError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to download video {youtube_id} after {max_retries} attempts: {e}")
                        return
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            # Process each clip
            for clip_id, clip_info in clips.items():
                # Skip if clip already exists
                clip_path = output_dir / f"{clip_id}.mp4"
                if clip_path.exists():
                    continue

                start_time = self._time_to_seconds(clip_info["start"])
                end_time = self._time_to_seconds(clip_info["finish"])

                # Verify clip length
                if abs(end_time - start_time - self.clip_length) > 0.1:  # Allow 0.1s tolerance
                    logger.warning(
                        f"Clip {clip_id} length mismatch: "
                        f"expected {self.clip_length}s, got {end_time - start_time}s"
                    )
                    continue

                temp_clip = output_dir / f"temp_{clip_id}.mp4"

                try:
                    # Extract clip with exact timing and fps
                    (
                        ffmpeg.input(str(temp_video), ss=start_time, t=self.clip_length)
                        .output(
                            str(temp_clip),
                            vf=f"fps={self.fps}",
                            acodec="aac",
                            vcodec="libx264",
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )

                    # Apply face cropping
                    crop_face_from_video(str(temp_clip), str(clip_path))

                    # Clean up temporary clip
                    temp_clip.unlink()

                except ffmpeg.Error as e:
                    logger.error(
                        f"Failed to process clip {clip_id}: {e.stderr.decode() if e.stderr else str(e)}"
                    )
                    continue

            # Clean up temporary video
            temp_video.unlink()

        except Exception as e:
            logger.error(f"Failed to process video {youtube_id}: {str(e)}")

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert time string (MM:SS) to seconds.

        Args:
            time_str: Time string in format "MM:SS"

        Returns:
            Time in seconds as float
        """
        minutes, seconds = map(float, time_str.split(":"))
        return minutes * 60 + seconds

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load dataset configuration from JSON file.

        Args:
            config_path: Optional path to config file. If None, uses environment variable.

        Returns:
            Dict containing dataset configuration.

        Raises:
            FileNotFoundError: If config file not found.
            JSONDecodeError: If config file is invalid JSON.
        """
        if config_path is None:
            config_path = get_env("DATASETS_CFG", "datasets_config.json")

        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            with open(config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def _load_samples(self) -> List[str]:
        """Load all samples from the dataset directory.

        Returns:
            List of dictionaries containing sample information:
                - path: Path to video file
                - metadata: Dictionary of metadata (if available)
        """

        def collect_files_recursively(root: str, samples_paths: list):
            for obj in os.scandir(root):
                if obj.is_dir():
                    collect_files_recursively(obj.path, samples_paths)
                elif obj.is_file() and obj.name.endswith(".mp4"):
                    samples_paths.append(obj.path)
                elif not obj.is_file():
                    logger.warning(f"Skipping non-file, non-directory: {obj.path}")

        samples = []
        collect_files_recursively(str(self.dataset_dir), samples)

        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset with optional transformation.

        Args:
            idx: Index of the sample to get.

        Returns:
            torch.Tensor: Video tensor in format [T, C, H, W].

        Raises:
            RuntimeError: If video loading or transformation fails.
            IndexError: If transform_in_use_idx is invalid.
        """
        try:
            # Get video path
            video_path = self.samples[idx]

            # Initialize base transforms
            to_bytes = ToBytes(target_fps=self.fps)
            to_tensor = ToTensor()

            # Check if additional transform should be applied
            add_transform = self._get_current_transform()

            # Apply transformation pipeline
            if add_transform is not None:
                video_bytes = to_bytes(video_path)
                video_bytes = add_transform(video_bytes)
                video = to_tensor(video_bytes)
            else:
                # Apply minimal pipeline
                video = to_tensor(to_bytes(video_path))

            return video

        except Exception as e:
            logger.error(
                f"Failed to process video {video_path} with transform "
                f"index {self.transform_in_use_idx}: {str(e)}"
            )
            raise RuntimeError(f"Video processing failed: {str(e)}")

    def _get_current_transform(self) -> Optional[Union[Rescale, Compress]]:
        """Get the current transform based on transform_in_use_idx.

        Returns:
            Optional[Union[Rescale, Compress]]: Transform instance or None if no transform
            should be applied.

        Raises:
            IndexError: If transform_in_use_idx is out of range.
            ValueError: If unknown transform type is specified.
        """
        if not self.transforms or self.transform_in_use_idx == -1:
            return None

        if not 0 <= self.transform_in_use_idx < len(self.transforms):
            raise IndexError(
                f"Transform index {self.transform_in_use_idx} is out of range "
                f"[0, {len(self.transforms)})"
            )

        transform_name, transform_value = self.transforms[self.transform_in_use_idx]

        # Map transform names to their classes
        transform_map = {
            "Rescale": lambda v: Rescale(float(v)),
            "Compress": lambda v: Compress(int(v)),
        }

        if transform_name not in transform_map:
            raise ValueError(f"Unknown transform type: {transform_name}")

        return transform_map[transform_name](transform_value)

    @property
    def current_transform_index(self) -> int:
        """Get current transform index.

        Returns:
            int: Current transform index (-1 if no transform is active)
        """
        return self.transform_in_use_idx

    @current_transform_index.setter
    def current_transform_index(self, idx: int) -> None:
        """Set current transform index.

        Args:
            idx: Transform index to use (-1 for no transform)

        Raises:
            IndexError: If idx is out of valid range
        """
        if idx != -1 and not 0 <= idx < len(self.transforms):
            raise IndexError(
                f"Transform index {idx} is out of range [-1, {len(self.transforms)})"
            )
        self.transform_in_use_idx = idx

    def _is_dataset_processed(self) -> bool:
        """Check if the dataset has been downloaded and processed.

        Returns:
            bool: True if dataset is fully processed, False otherwise
        """
        # Check if metadata exists
        sources_path = self.dataset_dir / "metadata" / "sources.json"
        if not sources_path.exists():
            logger.warning(
                f"Dataset {self.dataset_name} doesn't have metadata. Treating as processed."
            )
            return True

        try:
            # Load metadata to check expected files
            with open(sources_path) as f:
                sources = json.load(f)

            # Check both train and test splits
            for split in ["train", "test"]:
                split_dir = self.dataset_data_dir / split
                if not split_dir.exists():
                    return False

                # Check each identity directory
                for identity_id, identity_data in sources[split].items():
                    identity_dir = split_dir / identity_id
                    if not identity_dir.exists():
                        return False

                    # Check each video directory
                    for video_id, video_data in identity_data.items():
                        youtube_id = video_data["id"]
                        video_dir = identity_dir / youtube_id
                        if not video_dir.exists():
                            return False

                        # Check each clip file
                        for clip_id in video_data["clips"].keys():
                            clip_path = video_dir / f"{clip_id}.mp4"
                            if not clip_path.exists():
                                return False

            return True

        except Exception as e:
            logger.error(f"Error checking dataset status: {str(e)}")
            return False


def initialize_datasets() -> Dict[str, TMSDataset]:
    """Initialize all datasets from configuration.

    Returns:
        Dictionary of dataset name to TMSDataset instance.

    Raises:
        FileNotFoundError: If config file is not found.
        RuntimeError: If dataset initialization fails.
    """
    config_path = Path(get_env("DATASETS_CFG", "datasets_config.json"))
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    datasets = {}
    for dataset_name in config["datasets"]:
        try:
            datasets[dataset_name] = TMSDataset(dataset_name)
        except Exception as e:
            logger.error(f"Failed to initialize dataset {dataset_name}: {str(e)}")
            raise RuntimeError(f"Failed to initialize dataset {dataset_name}: {str(e)}")

    return datasets
