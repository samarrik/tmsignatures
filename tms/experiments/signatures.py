import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from tms.data.datasets import FacialFeaturesDataset, TMSDataset
from tms.data.utils.video_manipulations import unify_processed_video_file
from tms.experiments.models.resnet import ArcFaceResNet, ResNet, train_arcface_resnet


def signatures_distance(dataset: TMSDataset) -> None:
    """
    Calculate and visualize the distance between embeddings and class centroids.

    Args:
        dataset: The dataset to analyze
        models: List of models to use for the experiment
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

        # Setup the plot style
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

        # Use scientific style with no border on top and right
        sns.despine()

        # Save with publication quality
        plt.tight_layout()
        plt.savefig(figures_dir / f"distance_distribution_class_{target_label}.png", bbox_inches='tight')
        plt.close()


def contrastive_learning_finetuning(dataset: TMSDataset) -> None:
    """
    Fine-tune a ResNet model with ArcFace for contrastive learning.

    Args:
        dataset: The dataset to use for fine-tuning
        models: List of models to use for the experiment
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
        palette="husl",  # More professional color palette
        s=50,
        alpha=0.7,
        edgecolor="none",
    )

    # Customize the plot
    plt.title("t-SNE Visualization of ArcFace Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Identity", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Use scientific style with no border on top and right
    sns.despine()

    # Save with publication quality
    plt.tight_layout()
    plt.savefig(figures_dir / "arcface_embeddings_tsne.png", bbox_inches='tight')
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
        palette=["#2ecc71", "#e74c3c"],
    )

    # Customize the plot
    plt.title("ArcFace Fine-tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Use scientific style with no border on top and right
    sns.despine()

    # Save with publication quality
    plt.tight_layout()
    plt.savefig(figures_dir / "arcface_training_accuracy.png", bbox_inches='tight')
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
        models: List of models to use for the experiment
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

        # Setup the plot style
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

        # Use scientific style with no border on top and right
        sns.despine()

        # Save with publication quality
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"arcface_distance_distribution_class_{target_label}.png",
            bbox_inches='tight'
        )
        plt.close()


def deepfake_detection(
    talking_celebs_dataset: TMSDataset, additional_dataset: TMSDataset
) -> None:
    """
    Analyze embeddings from videos to detect deepfake regions by tracking distance to identity centroid.

    This experiment:
    1. Calculates Zelenskyi's centroid from TalkingCelebs training data
    2. Extracts embeddings from 30-second windows (15-second overlap) from Additional dataset videos
    3. Calculates distances to the centroid for each window
    4. Plots distances over time, highlighting deepfake regions

    Args:
        talking_celebs_dataset: The TalkingCelebs dataset containing Zelenskyi's training data
        additional_dataset: The dataset containing videos to analyze for deepfakes
    """
    print("Running deepfake detection experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ArcFace model
    model_path = (
        Path("models")
        / "experiments"
        / talking_celebs_dataset.name
        / "arcface_resnet_best.pth"
    )
    if not model_path.exists():
        print(
            "ArcFace model not found. Please run contrastive_learning_finetuning first."
        )
        return

    checkpoint = torch.load(model_path)
    model = ArcFaceResNet(num_classes=len(checkpoint["class_mapping"]))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Find all training files for Zelenskyi
    zelenskyi_id = "id00002"  # ID for Zelenskyi in TalkingCelebs
    train_processed_path = (
        talking_celebs_dataset.path
        / "processed"
        / "Raw"
        / "None"
        / "train"
        / zelenskyi_id
    )
    zelenskyi_files = list(train_processed_path.rglob("*.csv"))

    if not zelenskyi_files:
        print(f"No training data found for Zelenskyi ({zelenskyi_id})")
        return

    print(f"Found {len(zelenskyi_files)} training files for Zelenskyi")

    # Extract features and calculate centroid
    zelenskyi_embeddings = []

    for file_path in tqdm(zelenskyi_files, desc="Extracting Zelenskyi embeddings"):
        features_df = unify_processed_video_file(
            file_path,
            talking_celebs_dataset.features,
            talking_celebs_dataset.clip_length,
            talking_celebs_dataset.fps,
        )

        features = features_df.values.astype(np.float32)
        features_tensor = torch.tensor(features).to(device)

        with torch.no_grad():
            embedding = model.extract_embeddings(features_tensor.unsqueeze(0))
            embedding = embedding.cpu().numpy().flatten()
            zelenskyi_embeddings.append(embedding)

    zelenskyi_centroid = np.mean(np.stack(zelenskyi_embeddings), axis=0)

    # Create figures directory
    figures_dir = Path("results") / talking_celebs_dataset.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    video_info = [
        {
            "name": "fs_injected.mp4",
            "deepfake_start": 72,  # 1:12 in seconds
            "deepfake_end": 144,  # 2:24 in seconds
            "title": "FaceSwap Injected Video",
        },
        {
            "name": "ls_injected.mp4",
            "deepfake_start": 72,  # 1:12 in seconds
            "deepfake_end": 144,  # 2:24 in seconds
            "title": "LipSync Injected Video",
        },
        {
            "name": "long.mp4",
            "deepfake_start": None,
            "deepfake_end": None,
            "title": "Pristine Video (No Manipulation)",
        },
    ]

    # Window parameters
    window_size = 30  # seconds
    window_overlap = 15  # seconds

    for video in video_info:
        print(f"Processing {video['name']}...")

        processed_file = (
            additional_dataset.path
            / "processed"
            / "Raw"
            / "None"
            / video["name"].replace(".mp4", ".csv")
        )
        if not processed_file.exists():
            print(f"Processed file not found: {processed_file}")
            continue

        full_features = pd.read_csv(processed_file)
        full_features = full_features[talking_celebs_dataset.features]

        # Calculate windows
        frames_per_window = int(window_size * talking_celebs_dataset.fps)
        frames_step = int((window_size - window_overlap) * talking_celebs_dataset.fps)
        num_frames = len(full_features)
        window_starts = list(range(0, num_frames - frames_per_window + 1, frames_step))

        # Process each window
        window_times = []
        distances = []

        for start_idx in window_starts:
            end_idx = start_idx + frames_per_window
            window_features = full_features.iloc[start_idx:end_idx].values.astype(
                np.float32
            )
            window_center = (
                start_idx + frames_per_window / 2
            ) / talking_celebs_dataset.fps

            features_tensor = torch.tensor(window_features).to(device)

            with torch.no_grad():
                embedding = model.extract_embeddings(features_tensor.unsqueeze(0))
                embedding = embedding.cpu().numpy().flatten()
                distance = np.linalg.norm(embedding - zelenskyi_centroid)

            window_times.append(window_center)
            distances.append(distance)

        # Create plot with consistent style
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)
        plt.figure(figsize=(8, 5), dpi=150)

        plot_df = pd.DataFrame({"Time": window_times, "Distance": distances})

        # Create line plot with consistent styling
        sns.lineplot(
            data=plot_df,
            x="Time",
            y="Distance",
            markers=True,
            dashes=False,
            color="#2ecc71",
            linewidth=2,
            marker="o",
            markersize=6,
        )

        plt.ylim(0, 2)

        if video["deepfake_start"] is not None:
            plt.axvspan(
                video["deepfake_start"],
                video["deepfake_end"],
                alpha=0.3,
                color="#e74c3c",
                label="Deepfake Region",
            )

            plt.axvline(x=video["deepfake_start"], color="#e74c3c", linestyle="-", alpha=0.7, linewidth=1.5)
            plt.axvline(x=video["deepfake_end"], color="#e74c3c", linestyle="-", alpha=0.7, linewidth=1.5)

            mid_point = (video["deepfake_start"] + video["deepfake_end"]) / 2
            plt.text(
                mid_point,
                1.8,
                "DEEPFAKE",
                color="#c0392b",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round", pad=0.4, edgecolor="none"),
            )

        plt.title(f"Distance to Identity Centroid: {video['title']}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("L2 Distance")
        plt.grid(True, alpha=0.3)

        # Use scientific style with no border on top and right
        sns.despine()

        # Save with publication quality
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"deepfake_detection_{video['name'].replace('.mp4', '')}.png",
            bbox_inches='tight'
        )
        plt.close()
