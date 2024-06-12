import logging
import os

import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from .config import MODELS_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(processed_dir, method):
    feature_dir = os.path.join(processed_dir, method)
    logging.info(f"Loading data from {feature_dir}...")
    x_test = np.load(os.path.join(feature_dir, "test", "features.npy"))
    y_test = np.load(os.path.join(feature_dir, "test", "labels.npy"))
    return x_test, y_test


def evaluate_model(model, x_test, y_test, model_name, data_type):
    logging.info(f"Evaluating model {model_name} on {data_type} data...")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred)
    logging.info(f"Classification Report for {model_name} on {data_type}:\n{report}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")

    return {
        "model": model_name,
        "data_type": data_type,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": report,
    }


def main(processed_dir=PROCESSED_DATA_DIR, models_dir=MODELS_DIR, final=True):
    """
    Evaluate trained models on test data.

    Parameters:
    processed_dir (str): Directory containing processed data.
    models_dir (str): Directory containing trained models.
    final (bool): Evaluate final models.

    Returns:
    results (list): List of evaluation results for each model.
    """
    methods = ["flattened", "pca", "hog"]
    classifiers = ["LogisticRegression", "NaiveBayes", "SVM", "DummyClassifier"]
    results = []

    for method in methods:
        logging.info(f"Processing method: {method}")
        x_test, y_test = load_data(processed_dir, method)
        for clf_name in classifiers:
            logging.info(f"Processing classifier: {clf_name} for method: {method}")
            final_str = "_final" if final else ""
            model_path = os.path.join(models_dir, f"{clf_name}_{method}{final_str}.joblib")
            if os.path.exists(model_path):
                logging.info(f"Loading model from {model_path}")
                model = load(model_path)
                result = evaluate_model(model, x_test, y_test, clf_name, method)
                results.append(result)
            else:
                logging.warning(f"Model file {model_path} not found.")
        logging.info(f"Completed processing for method: {method}")

    return results


if __name__ == "__main__":
    results = main()
    for result in results:
        print(f"Model: {result['model']} on {result['data_type']} data")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(result["classification_report"])
        print("=" * 80)
