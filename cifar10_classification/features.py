import logging
import os

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_interim_data(interim_dir="data/interim"):
    logging.info("Loading interim data...")
    x_train = np.load(os.path.join(interim_dir, "x_train.npy"))
    y_train = np.load(os.path.join(interim_dir, "y_train.npy"))
    x_test = np.load(os.path.join(interim_dir, "x_test.npy"))
    y_test = np.load(os.path.join(interim_dir, "y_test.npy"))
    return x_train, y_train, x_test, y_test


def flatten_images(images):
    logging.info("Flattening images...")
    return images.reshape(images.shape[0], -1)


def extract_pca_features(images, n_components=50):
    logging.info(f"Extracting PCA features with {n_components} components...")
    pca = PCA(n_components=n_components)
    flattened_images = flatten_images(images)
    pca_features = pca.fit_transform(flattened_images)
    vr = pca.explained_variance_ratio_
    return pca_features, vr


def extract_hog_features(images):
    logging.info("Extracting HOG features...")
    features = [hog(image, channel_axis=-1) for image in images]
    return np.array(features)


def save_features(features, labels, method, processed_dir="data/processed", subset="train"):
    feature_dir = os.path.join(processed_dir, method, subset)
    logging.info(f"Saving features to {feature_dir}...")
    os.makedirs(feature_dir, exist_ok=True)
    np.save(os.path.join(feature_dir, "features.npy"), features)
    np.save(os.path.join(feature_dir, "labels.npy"), labels)


def extract_and_save_features(x_train, y_train, x_test, y_test, processed_dir="data/processed"):
    """
    Extract and save features from CIFAR-10 images.

    Parameters:
    x_train (numpy.ndarray): Training images.
    y_train (numpy.ndarray): Training labels.
    x_test (numpy.ndarray): Test images.
    y_test (numpy.ndarray): Test labels.
    processed_dir (str): Directory to save processed data.

    Returns:
    x_train_flat (numpy.ndarray): Flattened training features.
    x_train_pca (numpy.ndarray): PCA training features.
    vr_train (numpy.ndarray): Explained variance ratio of PCA features.
    x_train_hog (numpy.ndarray): HOG training features.
    x_test_flat (numpy.ndarray): Flattened test features.
    x_test_pca (numpy.ndarray): PCA test features.
    vr_test (numpy.ndarray): Explained variance ratio of PCA features.
    x_test_hog (numpy.ndarray): HOG test features.
    """
    # Flatten features
    x_train_flat = flatten_images(x_train)
    x_test_flat = flatten_images(x_test)
    save_features(x_train_flat, y_train, "flattened", processed_dir, "train")
    save_features(x_test_flat, y_test, "flattened", processed_dir, "test")

    # PCA features
    x_train_pca, vr_train = extract_pca_features(x_train)
    x_test_pca, vr_test = extract_pca_features(x_test)
    save_features(x_train_pca, y_train, "pca", processed_dir, "train")
    save_features(x_test_pca, y_test, "pca", processed_dir, "test")

    # HOG features
    x_train_hog = extract_hog_features(x_train)
    x_test_hog = extract_hog_features(x_test)
    save_features(x_train_hog, y_train, "hog", processed_dir, "train")
    save_features(x_test_hog, y_test, "hog", processed_dir, "test")

    logging.info("Feature extraction and saving process completed.")
    return (
        x_train_flat,
        x_train_pca,
        vr_train,
        x_train_hog,
        x_test_flat,
        x_test_pca,
        vr_test,
        x_test_hog,
    )
