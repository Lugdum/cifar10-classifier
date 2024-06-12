import pandas as pd

from cifar10_classification import config, metrics


def main():
    results = metrics.main(config.PROCESSED_DATA_DIR, config.MODELS_DIR, False)
    results_df = pd.DataFrame(results)
    print(results_df)
    print("Metrics evaluation complete.")

if __name__ == "__main__":
    main()
