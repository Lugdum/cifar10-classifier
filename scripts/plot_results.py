from cifar10_classification import plots, config

def main():
    plots.main(config.PROCESSED_DATA_DIR, config.MODELS_DIR, final=False)
    print("Plot generation complete.")

    model_path = config.MODELS_DIR / 'SVM_hog_final.joblib'
    plots.evaluate_single_model(model_path)
    print("Single model evaluation complete.")

if __name__ == "__main__":
    main()
