import subprocess
import sys

def run_pipeline():
    scripts = [
        "02_clean.py",
        "03_features.py",
        "04_train_baseline.py",
        "05_train_advanced.py",
        "06_visualize.py"
    ]
    
    print("=== STARTING BOOK PRICE PREDICTOR PIPELINE ===")
    for script in scripts:
        print(f"\n--- Running {script} ---")
        subprocess.run([sys.executable, script], check=True)
    print("\n=== PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_pipeline()