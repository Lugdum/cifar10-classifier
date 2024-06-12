import numpy as np

from cifar10_classification import config, visualize_features


def main():
    x_train_flat = np.load(config.PROCESSED_DATA_DIR / 'flattened' / 'train' / 'features.npy')
    y_train = np.load(config.PROCESSED_DATA_DIR / 'flattened' / 'train' / 'labels.npy')
    x_train_hog = np.load(config.PROCESSED_DATA_DIR / 'hog' / 'train' / 'features.npy')
    vr_train = np.load(config.PROCESSED_DATA_DIR / 'pca' / 'train' / 'explained_variance_ratio.npy')

    visualize_features.visualize_original_images(config.INTERIM_DATA_DIR, save=True)
    visualize_features.visualize_flattened(x_train_flat, y_train, save=True)
    visualize_features.visualize_hog(x_train_hog, y_train, save=True)
    visualize_features.visualize_pca(vr_train, save=True)
    print("Feature visualization complete.")

if __name__ == "__main__":
    main()
