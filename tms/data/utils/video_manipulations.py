import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN


def crop_talking_head(
    video_path: Union[str, Path],
    fps: int,
    resolution: Tuple[int, int],
    confidence_threshold: float = 0.9,
    batch_size: int = 32,
    position_smoothing: float = 0.9,
    size_smoothing: float = 0.85,
    detection_frequency: int = 1,  # How often to detect faces (1=every frame, 2=every other frame)
    output_path: Optional[Union[str, Path]] = None,  # Optional different output path
) -> None:
    """
    Crops the talking head from the video and saves it to a new file.

    Args:
        video_path: Path to the input video file
        fps: Target frames per second
        resolution: Target resolution as (width, height)
        confidence_threshold: Minimum confidence for face detection
        batch_size: Number of frames to process in one batch
        position_smoothing: Smoothing factor for position (0-1)
        size_smoothing: Smoothing factor for size (0-1)
        detection_frequency: How often to detect faces (1=every frame, 2=every other frame)
        output_path: Optional different output path
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path
    else:
        output_path = Path(output_path)

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize MTCNN with optimized parameters
    mtcnn = MTCNN(
        select_largest=True,
        device=device,
        post_process=False,
        keep_all=False,
    )

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary output file
    temp_path = output_path.with_suffix(".temp.mp4")

    # Initialize tracking variables
    last_valid_bbox = None
    last_valid_frame = None
    smoothed_center = None
    smoothed_size = None

    # Force detection frequency to be at least 1
    detection_frequency = max(1, detection_frequency)

    # Set up ffmpeg process with optimized parameters
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{resolution[0]}x{resolution[1]}",
            r=fps,
        )
        .output(str(temp_path), vcodec="libx264", pix_fmt="yuv420p", preset="ultrafast")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    def get_bbox_with_margin(
        frame_shape: Tuple[int, int, int], bbox: List[float], margin_factor: float = 1.3
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bbox with margin more efficiently.

        Args:
            frame_shape: Shape of the frame (height, width, channels)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            margin_factor: Factor to expand the bbox by

        Returns:
            Tuple of (x1, y1, x2, y2) with margin
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate expanded bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        new_size = max(x2 - x1, y2 - y1) * margin_factor

        # Constrain to frame bounds - calculate once and convert to int
        half_size = new_size / 2
        new_x1 = max(0, int(center_x - half_size))
        new_y1 = max(0, int(center_y - half_size))
        new_x2 = min(w, int(center_x + half_size))
        # Ensure height matches width for square crop
        new_y2 = min(h, new_y1 + (new_x2 - new_x1))

        return new_x1, new_y1, new_x2, new_y2

    def process_frame(
        frame: np.ndarray,
        boxes: Optional[np.ndarray],
        probs: Optional[np.ndarray],
        frame_idx: int,
    ) -> np.ndarray:
        """
        Process a single frame, detecting faces if needed based on frame index.

        Args:
            frame: Input frame
            boxes: Detected face boxes
            probs: Detection probabilities
            frame_idx: Index of the current frame

        Returns:
            Processed frame
        """
        nonlocal last_valid_bbox, last_valid_frame, smoothed_center, smoothed_size
        h, w = frame.shape[:2]

        should_detect = frame_idx % detection_frequency == 0
        has_detection = (
            boxes is not None
            and len(boxes) > 0
            and probs is not None
            and len(probs) > 0
        )

        if should_detect and has_detection and probs[0] >= confidence_threshold:
            # Get current bbox with margin
            x1, y1, x2, y2 = get_bbox_with_margin(frame.shape, boxes[0])
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            size = max(x2 - x1, y2 - y1)

            # Apply smoothing if we have previous detections
            if smoothed_center is not None:
                # Smooth center position and size in single statements
                new_center_x = (
                    position_smoothing * smoothed_center[0]
                    + (1 - position_smoothing) * center_x
                )
                new_center_y = (
                    position_smoothing * smoothed_center[1]
                    + (1 - position_smoothing) * center_y
                )
                new_size = size_smoothing * smoothed_size + (1 - size_smoothing) * size

                # Update smoothed values
                smoothed_center = (new_center_x, new_center_y)
                smoothed_size = new_size

                # Calculate final bbox with smoothed values
                half_size = smoothed_size / 2
                x1 = max(0, int(smoothed_center[0] - half_size))
                y1 = max(0, int(smoothed_center[1] - half_size))
                x2 = min(w, int(smoothed_center[0] + half_size))
                y2 = min(h, int(smoothed_center[1] + half_size))

                # Ensure square crop
                crop_size = min(x2 - x1, y2 - y1)
                x2 = x1 + crop_size
                y2 = y1 + crop_size
            else:
                # First detection, initialize smoothing
                smoothed_center = (center_x, center_y)
                smoothed_size = size

            # Update last valid detection
            last_valid_bbox = [x1, y1, x2, y2]
            last_valid_frame = frame.copy()

        elif last_valid_bbox is not None:
            # Use last valid bbox
            x1, y1, x2, y2 = last_valid_bbox

        else:
            # Default to center crop if no detection available
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 3
            x1 = center_x - size // 2
            y1 = center_y - size // 2
            x2 = center_x + size // 2
            y2 = center_y + size // 2

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Handle invalid crop dimensions
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
            # Fallback to center crop
            size = min(w, h) // 3
            x1 = max(0, (w - size) // 2)
            y1 = max(0, (h - size) // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)

        # Create crop and resize
        try:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:  # Empty crop
                # Fallback to full frame
                face_crop = frame

            # Resize to target resolution
            return cv2.resize(face_crop, resolution)
        except Exception as e:
            print(f"Error in frame {frame_idx}: {e}")
            # Return a black frame as fallback in case of error
            return np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    try:
        # Process frames in batches
        frame_idx = 0
        success = True  # Flag to track success

        # Simplify the batch processing algorithm
        while frame_idx < frame_count:
            # Read up to batch_size frames
            batch_frames = []
            for _ in range(min(batch_size, frame_count - frame_idx)):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            # Prepare indices for this batch
            batch_indices = list(range(frame_idx, frame_idx + len(batch_frames)))

            # Determine which frames need detection
            detection_indices = [
                i
                for i, idx in enumerate(batch_indices)
                if idx % detection_frequency == 0
            ]

            # Only run detection on frames that need it
            if detection_indices:
                detection_frames = [batch_frames[i] for i in detection_indices]
                # Convert to RGB for MTCNN
                rgb_detection_frames = [
                    cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in detection_frames
                ]

                # Run face detection
                boxes_batch, probs_batch = mtcnn.detect(rgb_detection_frames)

                # Map detection results back to original indices
                all_boxes = [None] * len(batch_frames)
                all_probs = [None] * len(batch_frames)

                for i, det_idx in enumerate(detection_indices):
                    all_boxes[det_idx] = boxes_batch[i]
                    all_probs[det_idx] = probs_batch[i]
            else:
                all_boxes = [None] * len(batch_frames)
                all_probs = [None] * len(batch_frames)

            # Process each frame
            for i, (frame, idx) in enumerate(zip(batch_frames, batch_indices)):
                boxes = all_boxes[i]
                probs = all_probs[i]

                processed_frame = process_frame(frame, boxes, probs, idx)
                process.stdin.write(processed_frame.tobytes())

            # Update frame index
            frame_idx += len(batch_frames)

    except Exception as e:
        print(f"Error during processing: {e}")
        success = False  # Set success to False if an error occurs

    finally:
        # Clean up
        cap.release()
        process.stdin.close()
        process.wait()

        # Replace original with processed file only if successful
        if success and temp_path.exists():
            temp_path.replace(output_path)


def to_bytes(sample: Union[str, Path]) -> Tuple[bytes, int, int]:
    """
    Converts a video file to raw bytes.

    Args:
        sample: Path to the video file

    Returns:
        Tuple of (raw_bytes, width, height)

    Raises:
        RuntimeError: If conversion fails
    """
    try:
        # Get video dimensions
        probe = ffmpeg.probe(sample)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        # Get raw bytes directly
        out, _ = (
            ffmpeg.input(sample)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )

        return out, width, height

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"Converting to bytes failed for {sample}: {error_msg}")


def to_tensor(video_bytes: Tuple[bytes, int, int]) -> torch.Tensor:
    """
    Converts raw video bytes to a PyTorch tensor.

    Args:
        video_bytes: Tuple of (raw_bytes, width, height)

    Returns:
        PyTorch tensor of shape (frames, channels, height, width)
    """
    video_data, width, height = video_bytes

    # More efficient tensor creation and reshape
    video = torch.frombuffer(video_data, dtype=torch.uint8).contiguous()
    video = video.view(-1, height, width, 3)  # Using view instead of reshape
    video = video.permute(
        0, 3, 1, 2
    ).contiguous()  # Make memory contiguous after permute

    return video


def rescale_video(
    video_bytes: Tuple[bytes, int, int], scale_coeff: float
) -> Tuple[bytes, int, int]:
    """
    Rescales a video by the given coefficient.

    Args:
        video_bytes: Tuple of (raw_bytes, width, height)
        scale_coeff: Scaling coefficient (0-1)

    Returns:
        Tuple of (raw_bytes, new_width, new_height)

    Raises:
        RuntimeError: If scaling fails
    """
    try:
        video_data, width, height = video_bytes

        # Calculate new dimensions (ensure they're even)
        new_width = (int(width * scale_coeff) // 2) * 2
        new_height = (int(height * scale_coeff) // 2) * 2

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

        # Optional: Save debug copy (comment out when in production)
        # debug_path = f'/home/samariva/tms/temp/rescale_{scale_coeff}.mp4'
        # (
        #     ffmpeg
        #     .input('pipe:',
        #         format='rawvideo',
        #         pix_fmt='rgb24',
        #         s=f'{new_width}x{new_height}'
        #     )
        #     .output(debug_path,
        #         vcodec='libx264',
        #         crf=23,
        #         preset='ultrafast'
        #     )
        #     .overwrite_output()
        #     .run(input=out, capture_stdout=True, capture_stderr=True)
        # )

        return out, new_width, new_height

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"Scaling failed: {error_msg}")


def compress_video(
    video_bytes: Tuple[bytes, int, int], crf_value: int
) -> Tuple[bytes, int, int]:
    """
    Compresses a video using the specified CRF value.

    Args:
        video_bytes: Tuple of (raw_bytes, width, height)
        crf_value: Constant Rate Factor value (0-51, lower is better quality)

    Returns:
        Tuple of (raw_bytes, width, height)

    Raises:
        RuntimeError: If compression fails
    """
    try:
        video_data, width, height = video_bytes

        # Use a temporary file with unique name
        temp_path = str(Path(f"temp/temp_{os.getpid()}_{time.time_ns()}.mp4"))
        Path(temp_path).parent.mkdir(parents=True, exist_ok=True)

        # Compress directly to h264 with optimized settings
        (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                framerate=25,
            )
            .output(
                temp_path,
                vcodec="libx264",
                crf=crf_value,
                preset="ultrafast",
                threads="auto",
            )
            .overwrite_output()
            .run(input=video_data, capture_stdout=True, capture_stderr=True)
        )

        # Optional: Save debug copy (comment out when in production)
        # debug_path = f'/home/samariva/tms/temp/compress_{crf_value}.mp4'
        # shutil.copy2(temp_path, debug_path)

        # Read back the compressed video
        out, _ = (
            ffmpeg.input(temp_path)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                threads="auto",
            )
            .run(capture_stdout=True, capture_stderr=True)
        )

        os.remove(temp_path)

        return out, width, height

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8") if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"Compression failed: {error_msg}")


def unify_processed_video_file(
    path: Union[str, Path], features: List[str], clip_length: int, fps: int
) -> pd.DataFrame:
    """
    Unifies the processed video file into a dataframe with the desired features.

    Args:
        path: Path to the processed video file (CSV)
        features: List of features to extract
        clip_length: Length of the clip in seconds
        fps: Frames per second

    Returns:
        DataFrame with unified features
    """
    # Unifies the processed video file into a dataframe with the desired features, clip length, and NaN handling
    df = pd.read_csv(path)
    df = df[features]

    # Substitute NaNs
    for col in df.columns:
        nan_ratio = df[col].isna().mean()
        df[col] = df[col].fillna(
            0 if nan_ratio > 0.2 else df[col].mean()  # 0.2 is the threshold for NaNs
        )

    if len(df) < clip_length * fps:
        # Pad the df
        df = df.reindex(range(clip_length * fps), fill_value=0)
    elif len(df) > clip_length * fps:
        # Truncate the df to the desired length by selecting a random starting point
        fcv = len(df)  # full clip video length
        fcc = int(clip_length * fps)  # full clip desired length
        clip_start = 0 if fcv == fcc else int(np.random.randint(0, fcv - fcc))
        df = df.iloc[clip_start : clip_start + fcc]

    return df
