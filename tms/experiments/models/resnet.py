import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tms.data.datasets import FacialFeaturesDataset


class ResNet(nn.Module):
    """ResNet model for feature extraction.

    This model adapts a pre-trained ResNet50 for single-channel input data
    and configures it for custom classification tasks.
    """

    def __init__(self, num_classes: int):
        """Initialize ResNet model.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()

        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Modify first conv layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final fc layer for our classification task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, height, width]

        Returns:
            Classification logits of shape [batch_size, num_classes]
        """
        x = x.unsqueeze(1)  # Add channel dimension
        return self.resnet(x)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the penultimate layer.

        Args:
            x: Input tensor of shape [batch_size, height, width]

        Returns:
            Normalized embeddings from the penultimate layer
        """
        x = x.unsqueeze(1)  # Add channel dimension

        # Process through ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        return torch.flatten(x, 1)  # Flatten before final fc layer


class ArcFaceResNet(nn.Module):
    """ResNet model combined with ArcFace for contrastive learning.

    This model uses a pre-trained ResNet as the feature extractor
    and adds an ArcFace layer for more discriminative embeddings.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_path: Optional[str] = None,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """Initialize ArcFaceResNet model.

        Args:
            num_classes: Number of output classes
            pretrained_path: Path to pretrained model weights
            scale: Scaling factor for logits
            margin: Angular margin for ArcFace
            easy_margin: Whether to use the easy margin variant
        """
        super().__init__()

        # Initialize the base ResNet model
        self.base_model = ResNet(num_classes=num_classes)

        # Load pre-trained weights if available
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

        # Get the embedding dimension from the base model
        embedding_dim = self._get_embedding_dimension()

        # ArcFace parameters
        self.in_features = embedding_dim
        self.out_features = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix (represents class centers)
        self.weight = nn.Parameter(
            torch.FloatTensor(self.out_features, self.in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin parameters
        self._precompute_margin_params()

    def _load_pretrained_weights(self, pretrained_path: Union[str, Path]) -> None:
        """Load pretrained weights for the base model.

        Args:
            pretrained_path: Path to the pretrained weights
        """
        checkpoint = torch.load(pretrained_path)
        self.base_model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded pre-trained model from {pretrained_path}")

    def _get_embedding_dimension(self) -> int:
        """Determine the embedding dimension using a dummy input.

        Returns:
            Dimension of the embedding vector
        """
        dummy_input = torch.rand(1, 128, 128)  # Standard input shape
        with torch.no_grad():
            return self.base_model.forward_embedding(dummy_input).shape[1]

    def _precompute_margin_params(self) -> None:
        """Precompute trigonometric values for the margin."""
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional ArcFace margin.

        Args:
            x: Input tensor of shape [batch_size, height, width]
            labels: Ground truth labels for ArcFace margin. If None,
                   returns scaled cosine similarities.

        Returns:
            Logits with ArcFace margin applied if labels provided
        """
        # Get embeddings from the base model
        embeddings = self.base_model.forward_embedding(x)

        # Normalize features and weights
        normalized_embeddings = F.normalize(embeddings)
        normalized_weights = F.normalize(self.weight)

        # Compute cosine similarity
        cosine = F.linear(normalized_embeddings, normalized_weights)

        # If no labels are provided, just return the scaled cosine similarity
        if labels is None:
            return cosine * self.scale

        # Apply ArcFace margin
        return self._apply_arcface_margin(cosine, labels)

    def _apply_arcface_margin(
        self, cosine: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply the ArcFace margin to cosine similarities.

        Args:
            cosine: Cosine similarities [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Logits with ArcFace margin applied
        """
        # Calculate sine based on cosine (Pythagorean identity)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Add angular margin: cos(θ+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin based on threshold
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for target classes
        one_hot = torch.zeros_like(cosine, device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin to target classes only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        return output * self.scale

    def get_logits_for_accuracy(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits for accuracy computation.

        Args:
            x: Input tensor of shape [batch_size, height, width]

        Returns:
            Scaled cosine similarities without margin
        """
        embeddings = self.base_model.forward_embedding(x)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        return torch.mm(normalized_embeddings, normalized_weights.t()) * self.scale

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embeddings from input.

        Args:
            x: Input tensor of shape [batch_size, height, width]

        Returns:
            Normalized embeddings
        """
        embeddings = self.base_model.forward_embedding(x)
        return F.normalize(embeddings, p=2, dim=1)

    @classmethod
    def from_pretrained(cls, num_classes: int, dataset_name: str) -> "ArcFaceResNet":
        """Create an ArcFaceResNet from pretrained weights.

        Args:
            num_classes: Number of output classes
            dataset_name: Name of the dataset used in the path

        Returns:
            Initialized ArcFaceResNet with pretrained weights
        """
        model_path = Path("models") / "experiments" / dataset_name / "resnet_best.pth"
        return cls(num_classes=num_classes, pretrained_path=model_path)


def train_resnet(
    train_dataset: FacialFeaturesDataset,
    test_dataset: FacialFeaturesDataset,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    num_workers: int = 1,  # Default to 1 for better process handling
    save_best_model: bool = False,
    early_stopping: bool = True,
) -> Dict[str, Any]:
    """Train a ResNet model on the given datasets.

    Args:
        train_dataset (FacialFeaturesDataset): Dataset for training
        test_dataset (FacialFeaturesDataset): Dataset for testing
        batch_size (int, optional): Batch size. Defaults to 32.
        num_epochs (int, optional): Number of epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        num_workers (int, optional): Number of workers. Defaults to 1.
        save_best_model (bool, optional): Whether to save the best model. Defaults to False.
        early_stopping (bool, optional): Whether to stop training when reaching 100% train accuracy. Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary containing the training and testing accuracies
    """
    # Ensure CUDA is properly initialized in the subprocess
    torch.cuda.empty_cache()

    # Initialize model
    model = ResNet(num_classes=len(train_dataset.identity_to_label))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # History of training metrics
    train_accuracies_h = []
    test_accuracies_h = []

    # Early stopping tracking
    early_stopped = False
    actual_epochs_run = 0
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
        if early_stopped:
            # If we've already reached 100% accuracy, just duplicate the last results
            train_accuracies_h.append(train_accuracies_h[-1])
            test_accuracies_h.append(test_accuracies_h[-1])
            continue

        actual_epochs_run += 1

        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_acc = 100.0 * correct / total
        train_accuracies_h.append(epoch_train_acc)

        # Test phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_test_acc = 100.0 * correct / total
        test_accuracies_h.append(epoch_test_acc)

        # Update best accuracies
        if epoch_train_acc > best_train_accuracy:
            best_train_accuracy = epoch_train_acc
            best_test_accuracy = epoch_test_acc

            # Save best model if requested
            if save_best_model:
                model_path = (
                    Path("models")
                    / "experiments"
                    / train_dataset.name
                    / "resnet_best.pth"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "class_mapping": train_dataset.identity_to_label,
                        "train_accuracy": epoch_train_acc,
                        "final_test_accuracy": epoch_test_acc,
                    },
                    model_path,
                )

        # Early stopping check
        if early_stopping and epoch_train_acc >= 100.0:
            print(
                f"Early stopping triggered at epoch {epoch + 1} with 100% training accuracy."
            )
            early_stopped = True

    # Save final model if not saving best model
    model_path = None
    if save_best_model:
        model_path = (
            Path("models") / "experiments" / train_dataset.name / "resnet_best.pth"
        )

    return {
        "train_accuracy": best_train_accuracy,
        "test_accuracy": best_test_accuracy,
        "train_accuracy_h": train_accuracies_h,
        "test_accuracy_h": test_accuracies_h,
        "path": str(model_path) if model_path else None,
        "early_stopped": early_stopped,
        "actual_epochs": actual_epochs_run,
    }


def train_arcface_resnet(
    train_dataset: FacialFeaturesDataset,
    test_dataset: FacialFeaturesDataset,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.0001,
    num_workers: int = 1,
    save_best_model: bool = False,
    early_stopping: bool = True,
    scale: float = 30.0,
    margin: float = 0.5,
    easy_margin: bool = False,
) -> Dict[str, Any]:
    """Train an ArcFace-enhanced ResNet model for contrastive learning.

    Args:
        train_dataset: Dataset for training
        test_dataset: Dataset for testing
        batch_size: Batch size for training
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        num_workers: Number of workers for data loading
        save_best_model: Whether to save the best model during training
        early_stopping: Whether to stop training when reaching 100% train accuracy
        scale: Scaling factor for ArcFace logits
        margin: Angular margin for ArcFace
        easy_margin: Whether to use the easy margin variant of ArcFace

    Returns:
        Dictionary containing training metrics and model path
    """
    # Ensure CUDA is properly initialized
    torch.cuda.empty_cache()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ArcFaceResNet.from_pretrained(
        num_classes=len(train_dataset.identity_to_label),
        dataset_name=train_dataset.name,
    )
    model = model.to(device)

    # Setup data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # History of training metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    all_preds = []
    all_labels = []

    # Training tracking variables
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    early_stopped = False
    actual_epochs_run = 0

    # Create directory for saving models
    if save_best_model:
        model_dir = Path("models") / "experiments" / train_dataset.name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "arcface_resnet_best.pth"
    else:
        model_path = None

    # Training loop
    for epoch in range(num_epochs):
        if early_stopped:
            # If we've already reached 100% accuracy, just duplicate the last results
            train_accuracies.append(train_accuracies[-1])
            test_accuracies.append(test_accuracies[-1])
            continue

        actual_epochs_run += 1
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass with labels for ArcFace
            outputs = model(features, labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics using logits without margin for accuracy calculation
            epoch_loss += loss.item()
            logits_for_accuracy = model.get_logits_for_accuracy(features)
            _, predicted = torch.max(logits_for_accuracy.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate epoch metrics
        epoch_train_acc = 100.0 * correct / total
        epoch_loss = epoch_loss / len(train_loader)

        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        temp_preds = []
        temp_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)

                # Use logits without margin for evaluation
                outputs = model.get_logits_for_accuracy(features)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store predictions and labels for confusion matrix
                temp_preds.extend(predicted.cpu().numpy())
                temp_labels.extend(labels.cpu().numpy())

        # Calculate test accuracy
        epoch_test_acc = 100.0 * correct / total
        test_accuracies.append(epoch_test_acc)

        # Update learning rate scheduler
        scheduler.step(epoch_train_acc)

        # Save the best model
        if epoch_train_acc > best_train_accuracy:
            best_train_accuracy = epoch_train_acc
            best_test_accuracy = epoch_test_acc
            all_preds = temp_preds
            all_labels = temp_labels

            if save_best_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_accuracy": epoch_train_acc,
                        "test_accuracy": epoch_test_acc,
                        "class_mapping": train_dataset.identity_to_label,
                    },
                    model_path,
                )

        # Early stopping if we reach 100% training accuracy
        if early_stopping and epoch_train_acc >= 100.0:
            early_stopped = True

    # Return training results
    return {
        "train_accuracy": best_train_accuracy,
        "test_accuracy": best_test_accuracy,
        "train_accuracy_h": train_accuracies,
        "test_accuracy_h": test_accuracies,
        "path": str(model_path) if model_path else None,
        "early_stopped": early_stopped,
        "actual_epochs": actual_epochs_run,
        "confusion_matrix_data": {"predictions": all_preds, "labels": all_labels},
    }
