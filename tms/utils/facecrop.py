import cv2
import torch
from facenet_pytorch import MTCNN


def crop_face_from_video(
    input_video_path: str,
    output_video_path: str,
    target_size: tuple = (224, 224),
    confidence_threshold: float = 0.9,
) -> None:
    """Crop the largest face from each frame of a video.

    Uses MTCNN face detection to crop faces, with smoothing between frames
    and fallback to previous valid crops when detection confidence is low.

    Args:
        input_video_path: Path to input video file.
        output_video_path: Path to save output video.
        target_size: Target resolution for face crop. Default: (224, 224).
        confidence_threshold: Minimum confidence for face detection. Default: 0.9.

    Raises:
        ValueError: If input video file cannot be opened.
    """
    # Initialize face detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
        select_largest=True, device=device, post_process=False, keep_all=False
    )

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    # Initialize tracking variables
    last_valid_bbox = None
    last_valid_frame = None
    smoothing_factor = 0.8

    # Process frames
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color space for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs = mtcnn.detect(frame_rgb)

        if boxes is not None and len(boxes) > 0 and probs[0] >= confidence_threshold:
            # Get largest face with high confidence
            bbox = boxes[0]

            # Add margin to bbox (VoxCeleb style)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox

            # Calculate expanded bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            bbox_width, bbox_height = x2 - x1, y2 - y1
            new_size = max(bbox_width, bbox_height) * 1.3

            # Constrain to frame bounds
            new_x1 = max(0, center_x - new_size / 2)
            new_y1 = max(0, center_y - new_size / 2)
            new_x2 = min(w, center_x + new_size / 2)
            new_y2 = min(h, center_y + new_size / 2)

            # Apply temporal smoothing
            if last_valid_bbox is not None:
                x1 = (
                    smoothing_factor * last_valid_bbox[0]
                    + (1 - smoothing_factor) * new_x1
                )
                y1 = (
                    smoothing_factor * last_valid_bbox[1]
                    + (1 - smoothing_factor) * new_y1
                )
                x2 = (
                    smoothing_factor * last_valid_bbox[2]
                    + (1 - smoothing_factor) * new_x2
                )
                y2 = (
                    smoothing_factor * last_valid_bbox[3]
                    + (1 - smoothing_factor) * new_y2
                )
            else:
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

            last_valid_bbox = [x1, y1, x2, y2]
            last_valid_frame = frame.copy()

        elif last_valid_bbox is not None:
            # Use last valid bbox
            x1, y1, x2, y2 = last_valid_bbox

        else:
            # Default to center crop
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 3
            x1 = center_x - size // 2
            y1 = center_y - size // 2
            x2 = center_x + size // 2
            y2 = center_y + size // 2

        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Get face crop
        if boxes is None or len(boxes) == 0 or probs[0] < confidence_threshold:
            face_crop = (
                last_valid_frame[y1:y2, x1:x2]
                if last_valid_frame is not None
                else frame[y1:y2, x1:x2]
            )
        else:
            face_crop = frame[y1:y2, x1:x2]

        # Resize and write frame
        face_resized = cv2.resize(face_crop, target_size)
        out.write(face_resized)

    # Clean up
    cap.release()
    out.release()
