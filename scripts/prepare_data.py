from cifar10_classification import dataset, config
from pathlib import Path

def main():
    x_train, y_train, x_test, y_test = dataset.prepare_data(Path(config.RAW_DATA_DIR) / 'cifar-10-batches-py', config.INTERIM_DATA_DIR)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
