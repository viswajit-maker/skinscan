"""Shared settings. Edit DATA_DIR if your dataset lives somewhere else."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "SkinDisNet_2"      # contains Preprocessed/, Augmented/, CSV
METADATA_CSV = DATA_DIR / "SkinDisNet_Metadata.csv"
IMAGE_DIR = DATA_DIR / "Preprocessed"                   # we train on ORIGINAL (preprocessed) images only

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLITS_CSV = OUTPUT_DIR / "splits.csv"
MODEL_PATH = OUTPUT_DIR / "best_model.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

HEAD_EPOCHS = 5          # phase 1: train only the new classifier layer
FINETUNE_EPOCHS = 30     # phase 2: train whole network
PATIENCE = 8             # early-stop if val macro-F1 doesn't improve for this many epochs
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
NUM_WORKERS = 2          # set to 0 if you get multiprocessing errors on Windows

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
