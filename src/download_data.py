from pathlib import Path
import kagglehub

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_is_present():
    """Return whether the output directory contains extracted image files."""
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in DATA_DIR.rglob("*")
    )

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    if dataset_is_present():
        print(f"Dataset already exists: {DATA_DIR}")
        return

    dataset_path = kagglehub.dataset_download(
        "mrgetshjtdone/vn-trash-classification",
        output_dir=str(DATA_DIR),
        force_download=True,
    )

    print(f"Dataset path: {dataset_path}")

if __name__ == "__main__":
    download_dataset()
