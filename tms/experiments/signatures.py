import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from tms.data.datasets import FacialFeaturesDataset, TMSDataset
from tms.experiments.models.resnet import ArcFaceResNet, ResNet, train_arcface_resnet


def signatures_distance(dataset: TMSDataset) -> None:
    """
    Calculate and visualize the distance between embeddings and class centroids.

    Args:
        dataset: The dataset to analyze
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    processed_loader = torch.utils.data.DataLoader(
        processed_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        pin_memory=True,
    )

    # Load the model
    model_path = Path("models") / "experiments" / dataset.name / "resnet_best.pth"
    checkpoint = torch.load(model_path)
    model = ResNet(num_classes=len(processed_dataset.identity_to_label))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Dictionary to store embeddings for each class
    embeddings_by_class = {}

    print("Extracting embeddings...")
    with torch.no_grad():
        for inputs, labels in processed_loader:
            inputs = inputs.to(device)
            outputs = model.forward_embedding(inputs)

            # Store embeddings by class
            for embedding, label in zip(outputs.cpu().numpy(), labels.numpy()):
                if label not in embeddings_by_class:
                    embeddings_by_class[label] = []
                embeddings_by_class[label].append(embedding)

    # Calculate centroids for each class
    print("Calculating centroids...")
    centroids = {}
    for label, embeddings in embeddings_by_class.items():
        centroids[label] = np.mean(embeddings, axis=0)

    # Create figures directory
    figures_dir = Path("results") / dataset.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Calculate distances and create histograms for each class
    for target_label in embeddings_by_class.keys():
        print(f"Processing class {target_label}...")

        # Calculate distances from centroids
        same_class_distances = []
        other_class_distances = []
        centroid = centroids[target_label]

        for label, embeddings in embeddings_by_class.items():
            distances = np.linalg.norm(np.array(embeddings) - centroid, axis=1)

            if label == target_label:
                same_class_distances.extend(distances)
            else:
                other_class_distances.extend(distances)

        # Prepare data for seaborn
        same_class_data = pd.DataFrame(
            {"Distance": same_class_distances, "Type": "Same Class"}
        )

        other_class_data = pd.DataFrame(
            {"Distance": other_class_distances, "Type": "Other Classes"}
        )

        df = pd.concat([same_class_data, other_class_data], ignore_index=True)

        # Setup the plot
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)
        plt.figure(figsize=(8, 5), dpi=150)

        # Create the histogram plot using seaborn
        sns.histplot(
            data=df,
            x="Distance",
            hue="Type",
            element="bars",
            stat="count",
            alpha=0.6,
            common_norm=False,
            palette=["#2ecc71", "#e74c3c"],
        )

        # Customize the plot
        plt.title(f"Distance Distribution for Class {target_label}")
        plt.xlabel("L2 Distance to Centroid")
        plt.ylabel("Count")
        plt.legend(title="")
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(figures_dir / f"distance_distribution_class_{target_label}.png")
        plt.close()


def contrastive_learning_finetuning(dataset: TMSDataset) -> None:
    """
    Fine-tune a ResNet model with ArcFace for contrastive learning.

    Args:
        dataset: The dataset to use for fine-tuning
    """
    print(f"\nRunning contrastive learning fine-tuning for {dataset.name}...")

    # Create train dataset
    train_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "train",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    # Create test dataset
    test_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    # Create directories for results
    figures_dir = Path("results") / dataset.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use the training function from resnet.py
    training_results = train_arcface_resnet(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_epochs=20,
        learning_rate=0.0001,
        weight_decay=0.0001,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        save_best_model=True,
        early_stopping=True,
    )

    # Extract training metrics from results
    train_accuracies = training_results["train_accuracy_h"]
    test_accuracies = training_results["test_accuracy_h"]
    best_train_accuracy = training_results["train_accuracy"]
    best_test_accuracy = training_results["test_accuracy"]
    early_stopped = training_results.get("early_stopped", False)
    actual_epochs_run = training_results.get("actual_epochs", len(train_accuracies))

    # Get confusion matrix data
    all_preds = training_results.get("confusion_matrix_data", {}).get("predictions", [])
    all_labels = training_results.get("confusion_matrix_data", {}).get("labels", [])

    print("\nFine-tuning completed!")
    print(f"Best train accuracy: {best_train_accuracy:.2f}%")
    print(f"Test accuracy for this model: {best_test_accuracy:.2f}%")
    if early_stopped:
        print(
            f"Note: Training stopped early after {actual_epochs_run} epochs due to 100% training accuracy."
        )

    # Extract embeddings for visualization
    print("Extracting embeddings for visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = (
        Path("models") / "experiments" / dataset.name / "arcface_resnet_best.pth"
    )
    checkpoint = torch.load(model_path)
    model = ArcFaceResNet(num_classes=len(train_dataset.identity_to_label))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Create data loader for test set
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        pin_memory=True,
    )

    # Extract embeddings
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)

            # Extract normalized embeddings
            embeddings = model.extract_embeddings(features)

            # Store embeddings and labels
            all_embeddings.extend(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # t-SNE visualization of embeddings
    print("Creating t-SNE visualization...")

    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Prepare data for seaborn
    tsne_data = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "Class": [f"Class {label}" for label in all_labels],
        }
    )

    # Create plot
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 8), dpi=150)

    # Create scatter plot using seaborn
    sns.scatterplot(
        data=tsne_data,
        x="x",
        y="y",
        hue="Class",
        palette="rainbow",
        s=50,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    # Customize the plot
    plt.title("t-SNE Visualization of ArcFace Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Identity", loc="best")

    # Save the plot
    plt.tight_layout()
    plt.savefig(figures_dir / "arcface_embeddings_tsne.png")
    plt.close()

    # Plot training metrics
    metrics_data = pd.DataFrame(
        {
            "Epoch": list(range(1, len(train_accuracies) + 1)),
            "Training Accuracy": train_accuracies,
            "Test Accuracy": test_accuracies,
        }
    )

    # Convert to long format for seaborn
    metrics_long = pd.melt(
        metrics_data,
        id_vars=["Epoch"],
        value_vars=["Training Accuracy", "Test Accuracy"],
        var_name="Metric",
        value_name="Accuracy",
    )

    # Create plot
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6), dpi=150)

    # Create line plot
    sns.lineplot(
        data=metrics_long,
        x="Epoch",
        y="Accuracy",
        hue="Metric",
        style="Metric",
        markers=True,
        dashes=False,
    )

    # Customize the plot
    plt.title("ArcFace Fine-tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(figures_dir / "arcface_training_accuracy.png")
    plt.close()

    # Create confusion matrix if data is available
    if all_preds and all_labels:
        print("Creating confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)

        # Get unique class labels
        unique_labels = np.unique(all_labels)

        # Create DataFrame for seaborn heatmap
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {i}" for i in range(len(unique_labels))],
            columns=[f"Pred {i}" for i in range(len(unique_labels))],
        )

        # Create plot
        sns.set_theme(style="white", context="paper", font_scale=1.2)
        plt.figure(figsize=(8, 6), dpi=150)

        # Create heatmap
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)

        # Customize the plot
        plt.title("Confusion Matrix for ArcFace Fine-tuned Model")
        plt.tight_layout()
        plt.savefig(figures_dir / "arcface_confusion_matrix.png")
        plt.close()


def arcface_signatures_distance(dataset: TMSDataset) -> None:
    """
    Analyze embeddings from the ArcFace fine-tuned model.

    Similar to signatures_distance, but uses the ArcFace model.
    Calculates distances from embeddings to class centroids and creates histograms.

    Args:
        dataset: The dataset to analyze
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test dataset
    test_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        pin_memory=True,
    )

    # Load the ArcFace fine-tuned model
    model_path = (
        Path("models") / "experiments" / dataset.name / "arcface_resnet_best.pth"
    )

    # Check if the model exists
    if not model_path.exists():
        print(
            f"ArcFace model not found at {model_path}. Please run contrastive_learning_finetuning first."
        )
        return

    checkpoint = torch.load(model_path)

    # Create ArcFace model and load state
    model = ArcFaceResNet(num_classes=len(test_dataset.identity_to_label))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Dictionary to store embeddings for each class
    embeddings_by_class = {}

    print("Extracting ArcFace embeddings...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Use extract_embeddings to get the normalized embeddings
            outputs = model.extract_embeddings(inputs)

            # Store embeddings by class
            for embedding, label in zip(outputs.cpu().numpy(), labels.numpy()):
                if label not in embeddings_by_class:
                    embeddings_by_class[label] = []
                embeddings_by_class[label].append(embedding)

    # Calculate centroids for each class
    print("Calculating centroids...")
    centroids = {}
    for label, embeddings in embeddings_by_class.items():
        centroids[label] = np.mean(embeddings, axis=0)

    # Create figures directory
    figures_dir = Path("results") / dataset.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Calculate distances and create histograms for each class
    for target_label in embeddings_by_class.keys():
        print(f"Processing class {target_label}...")

        # Calculate distances from centroids
        same_class_distances = []
        other_class_distances = []
        centroid = centroids[target_label]

        for label, embeddings in embeddings_by_class.items():
            distances = np.linalg.norm(np.array(embeddings) - centroid, axis=1)

            if label == target_label:
                same_class_distances.extend(distances)
            else:
                other_class_distances.extend(distances)

        # Prepare data for seaborn
        same_class_data = pd.DataFrame(
            {"Distance": same_class_distances, "Type": "Same Class"}
        )

        other_class_data = pd.DataFrame(
            {"Distance": other_class_distances, "Type": "Other Classes"}
        )

        df = pd.concat([same_class_data, other_class_data], ignore_index=True)

        # Setup the plot
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)
        plt.figure(figsize=(8, 5), dpi=150)

        # Create the histogram plot using seaborn
        sns.histplot(
            data=df,
            x="Distance",
            hue="Type",
            element="bars",
            stat="count",
            alpha=0.6,
            common_norm=False,
            palette=["#2ecc71", "#e74c3c"],
        )

        # Customize the plot
        plt.title(f"ArcFace Distance Distribution for Class {target_label}")
        plt.xlabel("L2 Distance to Centroid")
        plt.ylabel("Count")
        plt.legend(title="")
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"arcface_distance_distribution_class_{target_label}.png"
        )
        plt.close()
