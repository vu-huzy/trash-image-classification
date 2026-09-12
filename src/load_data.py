from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_data():
    image_paths = []
    labels = []

    for image_path in DATA_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            label = image_path.parent.name

            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels


if __name__ == "__main__":
    image_paths, labels = load_data()

    print(f"Total images: {len(image_paths)}")
    print(f"Total labels: {len(set(labels))}")
    print(f"Classes: {sorted(set(labels))}")