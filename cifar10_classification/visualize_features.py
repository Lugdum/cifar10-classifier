import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from .config import FIGURES_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(processed_dir, method, subset_size=None):
    feature_dir = os.path.join(processed_dir, method)
    logging.info(f"Loading data from {feature_dir}...")
    x_train = np.load(os.path.join(feature_dir, "train", "features.npy"))
    y_train = np.load(os.path.join(feature_dir, "train", "labels.npy"))
    x_test = np.load(os.path.join(feature_dir, "test", "features.npy"))
    y_test = np.load(os.path.join(feature_dir, "test", "labels.npy"))

    if subset_size:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_test = x_test[:subset_size]
        y_test = y_test[:subset_size]

    return x_train, y_train, x_test, y_test


def visualize_original_images(interim_dir=INTERIM_DATA_DIR, output_dir=FIGURES_DIR, save=False):
    """
    Visualize original CIFAR-10 images.

    Parameters:
    interim_dir (str): Directory containing interim data.
    output_dir (str): Directory to save output figures.
    save (bool): Save figures to disk.
    """
    x_train = np.load(os.path.join(interim_dir, "x_train.npy"))
    y_train = np.load(os.path.join(interim_dir, "y_train.npy"))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle("Original Images")
    for i in range(5):
        axes[i].imshow(x_train[i])
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "original_images.png"))
        plt.close()
    else:
        plt.show()


def visualize_flattened(x_train, y_train, output_dir=FIGURES_DIR, save=False):
    """
    Visualize flattened CIFAR-10 images.

    Parameters:
    x_train (numpy.ndarray): Flattened training features.
    y_train (numpy.ndarray): Training labels.
    output_dir (str): Directory to save output figures.
    save (bool): Save figures to disk.
    """
    x_train_reshaped = x_train.reshape(-1, 32, 32, 3)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle("Flattened Images")
    for i in range(5):
        axes[i].imshow(x_train_reshaped[i])
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "flattened_images.png"))
        plt.close()
    else:
        plt.show()


def visualize_pca(explained_variance_ratio, output_dir=FIGURES_DIR, save=False):
    """
    Visualize PCA explained variance ratio.

    Parameters:
    explained_variance_ratio (numpy.ndarray): Explained variance ratio of PCA features.
    output_dir (str): Directory to save output figures.
    save (bool): Save figures to disk.
    """
    explained_variance_cumsum = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(8, 6))
    plt.plot(explained_variance_cumsum)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "pca_variance.png"))
        plt.close()
    else:
        plt.show()


def visualize_hog(hog_features, y_train, output_dir=FIGURES_DIR, save=False):
    """
    Visualize HOG features.

    Parameters:
    hog_features (numpy.ndarray): HOG features.
    y_train (numpy.ndarray): Training labels.
    output_dir (str): Directory to save output figures.
    save (bool): Save figures to disk.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle("HOG Features")

    for i in range(5):
        hog_feature_length = hog_features.shape[1]

        width, height = 18, 18

        if width * height != hog_feature_length:
            print(
                f"Warning: HOG feature length {hog_feature_length} does not match expected size of {width * height}"
            )
            continue

        hog_image = hog_features[i].reshape(height, width)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        axes[i].imshow(hog_image_rescaled, cmap="gray")
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "hog_images.png"))
        plt.close()
    else:
        plt.show()


def main(processed_dir=PROCESSED_DATA_DIR, output_dir=FIGURES_DIR, save=False):
    # Visualize original images
    visualize_original_images(output_dir=output_dir, save=save)

    # Load and visualize flattened features
    x_train_flat, y_train_flat, _, _ = load_data(processed_dir, method="flattened")
    visualize_flattened(x_train_flat, y_train_flat, output_dir=output_dir, save=save)

    # Load and visualize PCA features
    explained_variance_ratio = np.load(
        os.path.join(processed_dir, "pca", "train", "explained_variance_ratio.npy")
    )
    visualize_pca(explained_variance_ratio, save=save)

    # Load HOG features and labels
    x_train_hog, y_train_hog, _, _ = load_data(processed_dir, method="hog")

    # Visualize HOG features
    visualize_hog(x_train_hog, y_train_hog, output_dir=output_dir, save=save)


if __name__ == "__main__":
    main(save=True)
