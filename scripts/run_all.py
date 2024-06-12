import subprocess

def main():
    scripts = [
        "prepare_data.py",
        "extract_features.py",
        "visualize_features.py",
        "train_models.py",
        "evaluate_metrics.py",
        "plot_results.py"
    ]

    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(["python3", f"scripts/{script}"])
        print(f"Completed {script}.\n")

if __name__ == "__main__":
    main()
