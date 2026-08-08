import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    scripts = [
        "02_clean.py",
        "09_advanced_features.py",
        "10_train_advanced_cv.py",
        "11_explainability.py"
    ]
    
    logger.info("=== STARTING ADVANCED BOOK PRICE PREDICTOR PIPELINE ===")
    for script in scripts:
        logger.info(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True)
    
    logger.info("Running pytest suite...")
    subprocess.run([sys.executable, "-m", "pytest"], check=True)
    logger.info("=== PIPELINE & TESTS EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_pipeline()