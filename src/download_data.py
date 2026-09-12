from pathlib import Path
import kagglehub

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        "mrgetshjtdone/vn-trash-classification",
        output_dir=str(DATA_DIR)
    )

    print(f"Dataset path: {dataset_path}")

if __name__ == "__main__":
    download_dataset()