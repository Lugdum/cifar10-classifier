import numpy as np
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from .config import MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(processed_dir, method, subset_size=None):
    feature_dir = os.path.join(processed_dir, method)
    logging.info(f"Loading data from {feature_dir}...")
    x_train = np.load(os.path.join(feature_dir, 'train', 'features.npy'))
    y_train = np.load(os.path.join(feature_dir, 'train', 'labels.npy'))
    x_test = np.load(os.path.join(feature_dir, 'test', 'features.npy'))
    y_test = np.load(os.path.join(feature_dir, 'test', 'labels.npy'))
    
    if subset_size:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_test = x_test[:subset_size]
        y_test = y_test[:subset_size]
    
    return x_train, y_train, x_test, y_test

def train_and_evaluate_model(model, param_distributions, x_train, y_train, x_test, y_test, model_name, data_type, use_hyperparameter_search):
    if use_hyperparameter_search:
        logging.info(f"Starting hyperparameter search for {model_name} on {data_type} data...")
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        logging.info(f"Best parameters for {model_name} on {data_type} data: {best_params}")
        best_model = random_search.best_estimator_
    else:
        logging.info(f"Training {model_name} on {data_type} data...")
        best_model = model
        best_model.fit(x_train, y_train)
        best_params = model.get_params()
    
    logging.info(f"Evaluating {model_name} on {data_type} data...")
    y_pred = best_model.predict(x_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    logging.info(f"Classification Report for {model_name} on {data_type}:\n{report}")
    logging.info(f"Confusion Matrix for {model_name} on {data_type}:\n{matrix}")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{data_type}.joblib")
    dump(best_model, model_path)
    logging.info(f"Model saved to {model_path}")

    return best_params

def re_train_with_best_params(model_class, best_params, x_train, y_train, model_name, data_type):
    logging.info(f"Re-training {model_name} on {data_type} data with best parameters...")
    model = model_class(**best_params)
    model.fit(x_train, y_train)
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{data_type}_final.joblib")
    dump(model, model_path)
    logging.info(f"Re-trained model saved to {model_path}")

def run_training(processed_dir='data/processed', use_hyperparameter_search=False, subset_size=None):
    """
    Train and evaluate models on different feature extraction methods.

    Parameters:
    processed_dir (str): Directory containing processed data.
    use_hyperparameter_search (bool): Use hyperparameter search for model training.
    subset_size (int): Number of samples to use for training and evaluation.
    """
    methods = ['flattened', 'pca', 'hog']
    classifiers = {
        'LogisticRegression': {
            'model_class': LogisticRegression,
            'model': LogisticRegression(max_iter=1000),
            'param_distributions': {
                'C': np.logspace(-4, 4, 20),
                'solver': ['liblinear', 'saga']
            }
        },
        'NaiveBayes': {
            'model_class': GaussianNB,
            'model': GaussianNB(),
            'param_distributions': {
                'var_smoothing': np.logspace(-9, -1, 20)
            }
        },
        'SVM': {
            'model_class': SVC,
            'model': SVC(),
            'param_distributions': {
                'C': np.logspace(-4, 4, 20),
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf']
            }
        },
        'DummyClassifier': {
            'model_class': DummyClassifier,
            'model': DummyClassifier(strategy='most_frequent'),
            'param_distributions': {
                'strategy': ['most_frequent', 'stratified', 'uniform', 'constant']
            }
        }
    }

    for method in methods:
        try:
            x_train, y_train, x_test, y_test = load_data(processed_dir, method, subset_size=subset_size)
            for clf_name, clf_info in classifiers.items():
                best_params = train_and_evaluate_model(clf_info['model'], clf_info['param_distributions'], x_train, y_train, x_test, y_test, clf_name, method, use_hyperparameter_search)
                
                if use_hyperparameter_search:
                    # Load full dataset for final training
                    x_train_full, y_train_full, x_test_full, y_test_full = load_data(processed_dir, method)
                    re_train_with_best_params(clf_info['model_class'], best_params, x_train_full, y_train_full, clf_name, method)
        except Exception as e:
            logging.error(f"Failed to process {method} data: {e}")

def re_train_all_with_best_params(processed_dir='data/processed'):
    methods = ['flattened', 'pca', 'hog']
    classifiers = {
        'LogisticRegression': LogisticRegression,
        'NaiveBayes': GaussianNB,
        'SVM': SVC,
        'DummyClassifier': DummyClassifier
    }

    for method in methods:
        try:
            x_train, y_train, x_test, y_test = load_data(processed_dir, method)
            for clf_name, clf_class in classifiers.items():
                model_path = f"models/{clf_name}_{method}.joblib"
                if os.path.exists(model_path):
                    best_model = load(model_path)
                    best_params = best_model.get_params()
                    re_train_with_best_params(clf_class, best_params, x_train, y_train, clf_name, method)
                else:
                    logging.error(f"Model file {model_path} not found.")
        except Exception as e:
            logging.error(f"Failed to process {method} data: {e}")
