
from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path("data/anime_faces/images")

image_files = [
    path for path in DATASET_DIR.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

print(f"Total image files: {len(image_files)}")

dimension_counts = Counter()
format_counts = Counter()
corrupted = []
sample_size = min(500, len(image_files))

for path in image_files[:sample_size]:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            dimension_counts[img.size] += 1
            format_counts[img.format] += 1
    except Exception as error:
        corrupted.append((str(path), str(error)))

print(f"\nImages inspected: {sample_size}")

print("\nMost common dimensions:")
for dimensions, count in dimension_counts.most_common(10):
    print(f"  {dimensions}: {count}")

print("\nImage formats:")
for image_format, count in format_counts.items():
    print(f"  {image_format}: {count}")

print(f"\nUnreadable images in inspected sample: {len(corrupted)}")

if corrupted:
    print("First few unreadable files:")
    for path, error in corrupted[:5]:
        print(f"  {path}: {error}")
