import multiprocessing as mp

from tms.data.datasets import initialize_datasets

mp.set_start_method("spawn", force=True)


def main():
    datasets = initialize_datasets()
    ...
    
    # Run experiments sequentially for each dataset
    for name, dataset in datasets.items():
        ...
        # sequence length
        # compression
        # rescale

    # signatures distance

    # contrastive learning finetuning

    # arcface signatures distance

    # longer video

    # injection attack


if __name__ == "__main__":
    main()
