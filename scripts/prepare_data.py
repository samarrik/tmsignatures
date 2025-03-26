import multiprocessing as mp

from tms.data.datasets import initialize_datasets

mp.set_start_method("spawn", force=True)


def main():
    datasets = initialize_datasets()

    for name, dataset in datasets.items():
        if not dataset.collected:
            dataset.collect()
        if not dataset.preprocessed:
            dataset.preprocess()
        if not dataset.processed:
            dataset.process()
        if not dataset.postprocessed:
            dataset.postprocess()


if __name__ == "__main__":
    main()
