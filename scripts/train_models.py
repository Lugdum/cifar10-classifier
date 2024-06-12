from cifar10_classification import train, config

def main():
    train.run_training(config.PROCESSED_DATA_DIR, use_hyperparameter_search=False, subset_size=1000)
    print("Model training complete.")

if __name__ == "__main__":
    main()
