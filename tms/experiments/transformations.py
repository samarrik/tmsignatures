import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from tms.data.datasets import FacialFeaturesDataset, TMSDataset
from tms.experiments.models.nsvm import train_nsvm
from tms.experiments.models.resnet import train_resnet
from tms.experiments.models.svm import train_svm


def visualize_and_save_results(
    dataset: TMSDataset,
    results: Dict[Union[float, int], Dict[str, Dict[str, float]]],
    name: str,
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    model_names: Dict[str, str],
) -> None:
    """
    Create and save a visualization of experiment results.

    Args:
        dataset: The dataset used in the experiment
        results: Dictionary mapping parameter values to model results
        name: Name of the experiment for file naming
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        legend_title: Title for the legend
        model_names: Dictionary mapping model keys to display names
    """
    # Create directories for results and figures
    results_data_dir = Path("results") / dataset.name / "data"
    results_figures_dir = Path("results") / dataset.name / "figures"
    results_data_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results as JSON
    path = results_data_dir / f"{name}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    # Set up plot style using seaborn
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 5), dpi=150)

    # Create DataFrame for seaborn
    plot_data = []
    x_values = sorted(results.keys())

    for x_val in x_values:
        for model in model_names:
            if model in results[x_val]:
                plot_data.append(
                    {
                        "Parameter": x_val,
                        "Model": model_names[model],
                        "Accuracy": results[x_val][model]["test_accuracy"],
                    }
                )

    df = pd.DataFrame(plot_data)

    # Create the line plot with consistent styling
    sns.lineplot(
        data=df,
        x="Parameter",
        y="Accuracy",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2,
        marker="o",
        markersize=6,
        palette=["#2ecc71", "#e74c3c", "#3498db"],  # Professional color scheme
    )

    # Customize the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Format x-axis for better readability
    if all(isinstance(x, float) for x in x_values):
        plt.xticks(x_values)

    # Use scientific style with no border on top and right
    sns.despine()

    # Save with publication quality
    plt.tight_layout()
    plt.savefig(
        results_figures_dir / f"{name}_plot.png",
        bbox_inches='tight'
    )
    plt.close()


def run_single_length(
    args: Tuple[TMSDataset, float],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Run experiments for a single sequence length.

    Args:
        args: Tuple containing (dataset, length_coefficient)

    Returns:
        Tuple of (length_coefficient, results dictionary)
    """
    dataset, length_coeff = args
    print(
        f"Training models for length_coeff={length_coeff} of the {dataset.name} dataset..."
    )

    # Load correlation data
    corr_test = pd.read_csv(dataset.path / "correlations" / f"{length_coeff}_test.csv")
    corr_train = pd.read_csv(
        dataset.path / "correlations" / f"{length_coeff}_train.csv"
    )

    # Filter correlation data
    corr_test_filtered = corr_test.query("compress == 0 & rescale == 0").drop(
        columns=["compress", "rescale"]
    )
    corr_train_filtered = corr_train.query("compress == 0 & rescale == 0").drop(
        columns=["compress", "rescale"]
    )

    # Prepare SVM data
    X_test = corr_test_filtered.drop("identity", axis=1)
    y_test = corr_test_filtered["identity"]
    X_train = corr_train_filtered.drop("identity", axis=1)
    y_train = corr_train_filtered["identity"]

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare ResNet data
    train_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "train",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )
    test_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Raw" / "None" / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    # Train models and return results
    return length_coeff, {
        "svm": train_svm(X_train, y_train, X_test, y_test),
        "nsvm": train_nsvm(X_train, y_train, X_test, y_test),
        "resnet": train_resnet(
            train_dataset,
            test_dataset,
            save_best_model=True if length_coeff == 1.0 else False,
            num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        ),
    }


def run_single_compression(
    args: Tuple[TMSDataset, int],
) -> Tuple[int, Dict[str, Dict[str, float]]]:
    """
    Run experiments for a single compression value.

    Args:
        args: Tuple containing (dataset, compression_value)

    Returns:
        Tuple of (compression_value, results dictionary)
    """
    dataset, compression_value = args
    print(
        f"Training models for compression={compression_value} of the {dataset.name} dataset..."
    )

    # Load correlation data
    corr_test = pd.read_csv(dataset.path / "correlations" / "1.0_test.csv")
    corr_train = pd.read_csv(dataset.path / "correlations" / "1.0_train.csv")

    # Filter correlation data
    corr_test_filtered = corr_test.query(
        f"compress == {compression_value} & rescale == 0"
    ).drop(columns=["compress", "rescale"])
    corr_train_filtered = corr_train.query(
        f"compress == {compression_value} & rescale == 0"
    ).drop(columns=["compress", "rescale"])

    # Prepare SVM data
    X_test = corr_test_filtered.drop("identity", axis=1)
    y_test = corr_test_filtered["identity"]
    X_train = corr_train_filtered.drop("identity", axis=1)
    y_train = corr_train_filtered["identity"]

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare ResNet data
    train_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path
        / "processed"
        / "Compress"
        / str(compression_value)
        / "train",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )
    test_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path
        / "processed"
        / "Compress"
        / str(compression_value)
        / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    # Train models and return results
    return compression_value, {
        "svm": train_svm(X_train, y_train, X_test, y_test),
        "nsvm": train_nsvm(X_train, y_train, X_test, y_test),
        "resnet": train_resnet(
            train_dataset,
            test_dataset,
            save_best_model=True if compression_value == 1.0 else False,
            num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        ),
    }


def run_single_rescale(
    args: Tuple[TMSDataset, float],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Run experiments for a single rescale value.

    Args:
        args: Tuple containing (dataset, rescale_coefficient)

    Returns:
        Tuple of (rescale_coefficient, results dictionary)
    """
    dataset, rescale_coeff = args
    print(
        f"Training models for rescale={rescale_coeff} of the {dataset.name} dataset..."
    )

    # Load correlation data
    corr_test = pd.read_csv(dataset.path / "correlations" / "1.0_test.csv")
    corr_train = pd.read_csv(dataset.path / "correlations" / "1.0_train.csv")

    # Filter correlation data
    corr_test_filtered = corr_test.query(
        f"compress == 0 & rescale == {rescale_coeff}"
    ).drop(columns=["compress", "rescale"])
    corr_train_filtered = corr_train.query(
        f"compress == 0 & rescale == {rescale_coeff}"
    ).drop(columns=["compress", "rescale"])

    # Prepare SVM data
    X_test = corr_test_filtered.drop("identity", axis=1)
    y_test = corr_test_filtered["identity"]
    X_train = corr_train_filtered.drop("identity", axis=1)
    y_train = corr_train_filtered["identity"]

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare ResNet data
    train_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Rescale" / str(rescale_coeff) / "train",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )
    test_dataset = FacialFeaturesDataset(
        name=dataset.name,
        root_path=dataset.path / "processed" / "Rescale" / str(rescale_coeff) / "test",
        features=dataset.features,
        clip_length=dataset.req_clip_length,
        fps=dataset.req_fps,
    )

    # Train models and return results
    return rescale_coeff, {
        "svm": train_svm(X_train, y_train, X_test, y_test),
        "nsvm": train_nsvm(X_train, y_train, X_test, y_test),
        "resnet": train_resnet(
            train_dataset,
            test_dataset,
            save_best_model=True if rescale_coeff == 1.0 else False,
            num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
        ),
    }


def sequence_length(dataset: TMSDataset, length_coeffs: List[float]) -> None:
    """
    Run experiments with different sequence lengths.

    Args:
        dataset: The dataset to use
        length_coeffs: List of length coefficients to test
    """
    print(f"\nRunning sequence length experiment for {dataset.name}...")

    # Create argument list for parallel processing
    args = [(dataset, coeff) for coeff in length_coeffs]

    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        results_list = list(executor.map(run_single_length, args))

    # Convert results list to dictionary
    results = dict(results_list)

    # Convert coefficients to actual sequence lengths in frames
    results = {
        float(coeff) * (dataset.req_clip_length * dataset.req_fps): results[coeff]
        for coeff in results.keys()
    }

    # Visualize and save results
    visualize_and_save_results(
        dataset=dataset,
        results=results,
        name="sequence_length",
        title="Effect of Sequence Length on Model Performance",
        x_label="Number of Frames",
        y_label="Test Accuracy (%)",
        legend_title="Models",
        model_names={
            "svm": "Support Vector Machine",
            "nsvm": "Multiclass Novelty SVM",
            "resnet": "ResNet50",
        },
    )


def compression(dataset: TMSDataset, compression_values: List[int]) -> None:
    """
    Run experiments with different compression values.

    Args:
        dataset: The dataset to use
        compression_values: List of compression values to test
    """
    print(f"\nRunning compression experiment for {dataset.name}...")

    # Create argument list for parallel processing
    args = [(dataset, value) for value in compression_values]

    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        results_list = list(executor.map(run_single_compression, args))

    # Convert results list to dictionary
    results = dict(results_list)

    # Visualize and save results
    visualize_and_save_results(
        dataset=dataset,
        results=results,
        name="compression",
        title="Effect of Video Compression on Model Performance",
        x_label="CRF Value (lower = better quality)",
        y_label="Test Accuracy (%)",
        legend_title="Models",
        model_names={
            "svm": "Support Vector Machine",
            "nsvm": "Multiclass Novelty SVM",
            "resnet": "ResNet50",
        },
    )


def rescale(dataset: TMSDataset, rescale_coeffs: List[float]) -> None:
    """
    Run experiments with different rescale coefficients.

    Args:
        dataset: The dataset to use
        rescale_coeffs: List of rescale coefficients to test
    """
    print(f"\nRunning rescale experiment for {dataset.name}...")

    # Create argument list for parallel processing
    args = [(dataset, coeff) for coeff in rescale_coeffs]

    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        results_list = list(executor.map(run_single_rescale, args))

    # Convert results list to dictionary
    results = dict(results_list)

    # Convert coefficients to actual frame dimensions
    results = {
        float(coeff) * dataset.req_resolution[0]: results[coeff]
        for coeff in results.keys()
    }

    # Visualize and save results
    visualize_and_save_results(
        dataset=dataset,
        results=results,
        name="resolution",
        title="Effect of Video Resolution on Model Performance",
        x_label="Width in Pixels",
        y_label="Test Accuracy (%)",
        legend_title="Models",
        model_names={
            "svm": "Support Vector Machine",
            "nsvm": "Multiclass Novelty SVM",
            "resnet": "ResNet50",
        },
    )
