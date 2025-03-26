"""This is a reduced version of the original PY-FEAT detector.py logic.

Original PY-FEAT detector.py file: https://github.com/cosanlab/py-feat/blob/main/feat/detector.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from feat.au_detectors.StatLearning.SL_test import XGBClassifier
from feat.data import TensorDataset
from feat.facepose_detectors.img2pose.deps.models import (
    FasterDoFRCNN,
    postprocess_img2pose,
)
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.pretrained import AU_LANDMARK_MAP, load_model_weights
from feat.utils import (
    FEAT_FACEPOSE_COLUMNS_6D,
)
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
    extract_hog_features,
)
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from skops.io import get_untrusted_types, load
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm


class Detector(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, au_model="xgb", pose_model="img2pose", landmark_model="mobilefacenet"
    ):
        super().__init__()

        if (
            au_model != "xgb"
            or pose_model != "img2pose"
            or landmark_model != "mobilefacenet"
        ):
            raise ValueError(
                "Only xgb, img2pose and mobilefacenet models are supported"
            )

        self.au_model = "xgb"
        self.pose_model = "img2pose"
        self.landmark_model = "mobilefacenet"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.info = {
            "facepose_model": "img2pose",
            "landmark_model": "mobilefacenet",
            "au_model": "xgb",
        }

        self._init_facepose_detector()
        self._init_landmark_detector()
        self._init_au_detector()

    def _init_facepose_detector(self):
        facepose_config_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="config.json",
            cache_dir=Path("models/detector"),
        )
        with open(facepose_config_file, "r") as f:
            facepose_config = json.load(f)

        # Initialize img2pose
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
        backbone.eval()
        backbone.to(self.device)
        self.facepose_detector = FasterDoFRCNN(
            backbone=backbone,
            num_classes=2,
            min_size=facepose_config["min_size"],
            max_size=facepose_config["max_size"],
            pose_mean=torch.tensor(facepose_config["pose_mean"]),
            pose_stddev=torch.tensor(facepose_config["pose_stddev"]),
            threed_68_points=torch.tensor(facepose_config["threed_points"]),
            rpn_pre_nms_top_n_test=facepose_config["rpn_pre_nms_top_n_test"],
            rpn_post_nms_top_n_test=facepose_config["rpn_post_nms_top_n_test"],
            bbox_x_factor=facepose_config["bbox_x_factor"],
            bbox_y_factor=facepose_config["bbox_y_factor"],
            expand_forehead=facepose_config["expand_forehead"],
        )
        facepose_model_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="model.safetensors",
            cache_dir=Path("models/detector"),
        )
        facepose_checkpoint = load_file(facepose_model_file)
        self.facepose_detector.load_state_dict(facepose_checkpoint, load_model_weights)
        self.facepose_detector.eval()
        self.facepose_detector.to(self.device)
        # self.facepose_detector = torch.compile(self.facepose_detector)

    def _init_landmark_detector(self):
        self.face_size = 112
        self.landmark_detector = MobileFaceNet(
            [self.face_size, self.face_size], 136, device=self.device
        )
        landmark_model_file = hf_hub_download(
            repo_id="py-feat/mobilefacenet",
            filename="mobilefacenet_model_best.pth.tar",
            cache_dir=Path("models/detector"),
        )
        landmark_state_dict = torch.load(
            landmark_model_file, map_location=self.device, weights_only=True
        )["state_dict"]  # Ensure Model weights are Float32 for MPS
        self.landmark_detector.load_state_dict(landmark_state_dict)
        self.landmark_detector.to(self.device)
        self.landmark_detector.eval()

    def _init_au_detector(self):
        self.au_detector = XGBClassifier()
        au_model_path = hf_hub_download(
            repo_id="py-feat/xgb_au",
            filename="xgb_au_classifier.skops",
            cache_dir=Path("models/detector"),
        )
        au_unknown_types = get_untrusted_types(file=au_model_path)
        loaded_au_model = load(au_model_path, trusted=au_unknown_types)
        self.au_detector.load_weights(
            scaler_upper=loaded_au_model.scaler_upper,
            pca_model_upper=loaded_au_model.pca_model_upper,
            scaler_lower=loaded_au_model.scaler_lower,
            pca_model_lower=loaded_au_model.pca_model_lower,
            scaler_full=loaded_au_model.scaler_full,
            pca_model_full=loaded_au_model.pca_model_full,
            classifiers=loaded_au_model.classifiers,
        )

    @torch.inference_mode()
    def detect_faces(self, images, face_size=112, face_detection_threshold=0.5):
        # img2pose
        frames = convert_image_to_tensor(images, img_type="float32") / 255.0
        frames.to(self.device)

        batch_results = []
        for i in range(frames.size(0)):
            single_frame = frames[i, ...].unsqueeze(
                0
            )  # Extract single image from batch
            img2pose_output = self.facepose_detector(single_frame.to(self.device))
            img2pose_output = postprocess_img2pose(
                img2pose_output[0], detection_threshold=face_detection_threshold
            )
            bbox = img2pose_output["boxes"]
            poses = img2pose_output["dofs"]
            facescores = img2pose_output["scores"]

            # Extract faces from bbox
            if bbox.numel() != 0:
                extracted_faces, new_bbox = extract_face_from_bbox_torch(
                    single_frame, bbox, face_size=face_size
                )
            else:  # No Face Detected - let's test of nans will work
                extracted_faces = torch.zeros((1, 3, face_size, face_size))
                bbox = torch.full((1, 4), float("nan"))
                new_bbox = torch.full((1, 4), float("nan"))
                facescores = torch.zeros((1))
                poses = torch.full((1, 6), float("nan"))

            frame_results = {
                "face_id": i,
                "faces": extracted_faces,
                "boxes": bbox,
                "new_boxes": new_bbox,
                "poses": poses,
                "scores": facescores,
            }

            batch_results.append(frame_results)

        return batch_results

    @torch.inference_mode()
    def forward(self, faces_data):
        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)

        # mobilefacenet
        landmarks = self.landmark_detector.forward(extracted_faces.to(self.device))[0]

        hog_features, au_new_landmarks = extract_hog_features(
            extracted_faces, landmarks
        )
        aus = self.au_detector.detect_au(
            frame=hog_features, landmarks=[au_new_landmarks]
        )

        poses = torch.cat(
            [face_output["poses"].to(self.device) for face_output in faces_data], dim=0
        )
        poses_np = poses.cpu().detach().numpy()

        # Check if poses data contains NaN values (empty data)
        has_empty_poses = np.isnan(poses_np).all(axis=1)

        # Could be optimized to:
        all_data = np.hstack([poses_np, aus])
        all_columns = FEAT_FACEPOSE_COLUMNS_6D + AU_LANDMARK_MAP["Feat"]
        feat_data = pd.DataFrame(all_data, columns=all_columns)

        # Handle NaN values directly on the combined DataFrame
        if has_empty_poses.any():
            mask = has_empty_poses
            feat_data.loc[mask, AU_LANDMARK_MAP["Feat"]] = np.nan

        return feat_data

    def detect(
        self,
        inputs,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_detection_threshold=0.5,
        progress_bar=True,
        save=None,
    ):
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("Only tensor inputs are supported")

        save_path = save if save else None
        data_loader = DataLoader(
            TensorDataset(inputs),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        results = []
        for batch in tqdm(data_loader) if progress_bar else data_loader:
            batch_results = self.forward(
                self.detect_faces(
                    batch["Image"],
                    face_size=self.face_size,
                    face_detection_threshold=face_detection_threshold,
                )
            )

            if save_path:
                batch_results.to_csv(
                    save_path, mode="a", index=False, header=not results
                )
            else:
                results.append(batch_results)

        if save_path:
            final_results = pd.read_csv(save_path)
            final_results.to_csv(save_path, mode="w", index=False)
            return final_results
        else:
            final_results = pd.concat(results).reset_index(drop=True)
            final_results.attrs.update(results[0].attrs)
            return final_results
