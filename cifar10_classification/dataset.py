import logging
import os
import pickle

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_batch(filepath):
    try:
        with open(filepath, "rb") as f:
            dict = pickle.load(f, encoding="bytes")
        return dict[b"data"], dict[b"labels"]
    except Exception as e:
        logging.error(f"Failed to load batch from {filepath}: {e}")
        raise


def load_data(data_dir="data/raw/cifar-10-batches-py"):
    x_train = []
    y_train = []
    for i in range(1, 6):
        filepath = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_batch(filepath)
        x_train.append(data)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    filepath = os.path.join(data_dir, "test_batch")
    x_test, y_test = load_batch(filepath)

    x_train = x_train.reshape((len(x_train), 3, 32, 32)).transpose(0, 2, 3, 1)
    x_test = x_test.reshape((len(x_test), 3, 32, 32)).transpose(0, 2, 3, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x, y):
    x = x.astype("float32") / 255.0
    y = np.squeeze(y)
    return x, y


def save_interim_data(x_train, y_train, x_test, y_test, interim_dir="data/interim"):
    os.makedirs(interim_dir, exist_ok=True)
    np.save(os.path.join(interim_dir, "x_train.npy"), x_train)
    np.save(os.path.join(interim_dir, "y_train.npy"), y_train)
    np.save(os.path.join(interim_dir, "x_test.npy"), x_test)
    np.save(os.path.join(interim_dir, "y_test.npy"), y_test)


def prepare_data(raw_data_dir="data/raw/cifar-10-batches-py", interim_dir="data/interim"):
    """
    Load and preprocess CIFAR-10 data.

    Parameters:
    raw_data_dir (str): Directory containing raw CIFAR-10 data.
    interim_dir (str): Directory to save interim data.

    Returns:
    x_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    x_test (np.ndarray): Test features.
    y_test (np.ndarray): Test labels.
    """
    logging.info("Loading and preprocessing data...")
    try:
        (x_train, y_train), (x_test, y_test) = load_data(raw_data_dir)
        x_train, y_train = preprocess_data(x_train, y_train)
        x_test, y_test = preprocess_data(x_test, y_test)
        save_interim_data(x_train, y_train, x_test, y_test, interim_dir)
        logging.info("Data preparation completed.")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        raise
