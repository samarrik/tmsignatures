import multiprocessing as mp

from tms.data.datasets import initialize_datasets
from tms.experiments.signatures import (
    arcface_signatures_distance,
    contrastive_learning_finetuning,
    deepfake_detection,
    signatures_distance,
)
from tms.experiments.transformations import (
    compression,
    rescale,
    sequence_length,
)

mp.set_start_method("spawn", force=True)


def main():
    # Initialize all datasets
    datasets = initialize_datasets()

    # Run transformation experiments for each dataset
    for name, dataset in datasets.items():
        print(f"\nRunning transformation experiments for {name} dataset...")

        # Sequence length experiment
        sequence_length(
            dataset=dataset, length_coeffs=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
        )

        # Compression experiment
        compression(dataset=dataset, compression_values=[28, 36, 41, 51])

        # Rescale experiment
        rescale(dataset=dataset, rescale_coeffs=[0.75, 0.5, 0.25, 0.1])

    # Run signature experiments for each dataset
    for name, dataset in datasets.items():
        print(f"\nRunning signature experiments for {name} dataset...")

        # Calculate distances to centroids
        signatures_distance(dataset)

        # Fine-tune with contrastive learning
        contrastive_learning_finetuning(dataset)

        # Calculate distances with ArcFace
        arcface_signatures_distance(dataset)

    # Run deepfake detection experiment
    if "TalkingCelebs" in datasets and "Additional" in datasets:
        print("\nRunning deepfake detection experiment...")
        deepfake_detection(
            talking_celebs_dataset=datasets["TalkingCelebs"],
            additional_dataset=datasets["Additional"],
        )
    else:
        print("\nSkipping deepfake detection experiment - required datasets not found")


if __name__ == "__main__":
    main()
