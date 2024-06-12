import numpy as np

from cifar10_classification import config, features


def main():
    x_train = np.load(config.INTERIM_DATA_DIR / 'x_train.npy')
    y_train = np.load(config.INTERIM_DATA_DIR / 'y_train.npy')
    x_test = np.load(config.INTERIM_DATA_DIR / 'x_test.npy')
    y_test = np.load(config.INTERIM_DATA_DIR / 'y_test.npy')

    x_train_flat, x_train_pca, vr_train, x_train_hog, x_test_flat, x_test_pca, vr_test, x_test_hog = features.extract_and_save_features(x_train, y_train, x_test, y_test, config.PROCESSED_DATA_DIR)
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()
