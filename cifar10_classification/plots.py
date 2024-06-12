import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from joblib import load
import logging
from .config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(processed_dir, method):
    feature_dir = os.path.join(processed_dir, method)
    logging.info(f"Loading data from {feature_dir}...")
    x_test = np.load(os.path.join(feature_dir, 'test', 'features.npy'))
    y_test = np.load(os.path.join(feature_dir, 'test', 'labels.npy'))
    return x_test, y_test

def plot_roc_curves_subplot(ax, y_test, y_score, model_name, data_type):
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_name}')
    ax.legend(loc="lower right")

def plot_combined_roc_curves(models, x_test, y_test, data_type, output_dir=None):
    logging.info(f"Generating combined ROC curves for {data_type} data...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    y_test_bin = label_binarize(y_test, classes=np.arange(y_test.max() + 1))

    for ax, (model_name, model) in zip(axes, models.items()):
        y_score = model.decision_function(x_test) if hasattr(model, "decision_function") else model.predict_proba(x_test)
        if y_score.ndim == 1:
            y_score = y_score[:, np.newaxis]
        plot_roc_curves_subplot(ax, y_test_bin, y_score, model_name, data_type)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'combined_roc_curves_{data_type}.png'))
        logging.info(f"ROC curves saved to {os.path.join(output_dir, f'combined_roc_curves_{data_type}.png')}")
        plt.close()
    else:
        plt.show()

def plot_combined_confusion_matrices(models, x_test, y_test, data_type, output_dir=None):
    logging.info(f"Generating combined confusion matrices for {data_type} data...")
    fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
    fig.suptitle(f'Combined Confusion Matrices for {data_type}', fontsize=16)

    for ax, (model_name, model) in zip(axes, models.items()):
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.set_title(f'{model_name}')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'combined_confusion_matrices_{data_type}.png'))
        logging.info(f"Confusion matrices saved to {os.path.join(output_dir, f'combined_confusion_matrices_{data_type}.png')}")
        plt.close()
    else:
        plt.show()

def evaluate_models(models, x_test, y_test, data_type, output_dir=None):
    plot_combined_roc_curves(models, x_test, y_test, data_type, output_dir)
    plot_combined_confusion_matrices(models, x_test, y_test, data_type, output_dir)

def main(processed_dir=PROCESSED_DATA_DIR, models_dir=MODELS_DIR, output_dir=FIGURES_DIR, final=True):
    """
    Evaluate trained models on test data.

    Parameters:
    processed_dir (str): Directory containing processed data.
    models_dir (str): Directory containing trained models.
    output_dir (str): Directory to save output figures.
    final (bool): Evaluate final models.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    methods = ['flattened', 'pca', 'hog']
    classifiers = ['LogisticRegression', 'NaiveBayes', 'SVM', 'DummyClassifier']

    for method in methods:
        logging.info(f"Processing method: {method}")
        x_test, y_test = load_data(processed_dir, method)
        models = {}
        for clf_name in classifiers:
            logging.info(f"Processing classifier: {clf_name} for method: {method}")
            final_str = "_final" if final else ""
            model_path = os.path.join(models_dir, f"{clf_name}_{method}{final_str}.joblib")
            if os.path.exists(model_path):
                logging.info(f"Loading model from {model_path}")
                models[clf_name] = load(model_path)
            else:
                logging.warning(f"Model file {model_path} not found.")
        if models:
            evaluate_models(models, x_test, y_test, method, output_dir)
        logging.info(f"Completed processing for method: {method}")

def evaluate_single_model(model_path, processed_dir=PROCESSED_DATA_DIR, output_dir=FIGURES_DIR):
    """
    Evaluate a single model on test data.

    Parameters:
    model_path (str): Path to the model file.
    processed_dir (str): Directory containing processed data.
    output_dir (str): Directory to save output figures.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        return
    
    logging.info(f"Loading model from {model_path}")
    model = load(model_path)
    # Extract the model name and data type from the file name
    base_name = os.path.basename(model_path)
    parts = base_name.split('_')
    model_name = parts[0]
    data_type = parts[1] if len(parts) > 2 else parts[1].split('.joblib')[0]

    x_test, y_test = load_data(processed_dir, data_type)
    
    # Evaluate single model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ROC curve
    y_score = model.decision_function(x_test) if hasattr(model, "decision_function") else model.predict_proba(x_test)
    n_classes = y_score.shape[1]
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:0.2f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve for {model_name} on {data_type}')
    axes[0].legend(loc="lower right")

    # Plot Confusion Matrix
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1])
    axes[1].set_title(f'Confusion Matrix for {model_name} on {data_type}')

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'evaluation_{model_name}_{data_type}.png'))
        logging.info(f"Evaluation plot saved to {os.path.join(output_dir, f'evaluation_{model_name}_{data_type}.png')}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
