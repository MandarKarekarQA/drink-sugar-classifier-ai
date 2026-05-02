from pathlib import Path
from PIL import Image

DATA_DIR = Path("Data")

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

bad_files = []

for file_path in DATA_DIR.rglob("*"):
    if file_path.is_file():
        try:
            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                img.convert("RGB").save(file_path.with_suffix(".jpg"), "JPEG")

            if file_path.suffix.lower() != ".jpg":
                file_path.unlink()

        except Exception:
            bad_files.append(file_path)

print(f"Bad files found: {len(bad_files)}")

for bad_file in bad_files:
    print("Deleting:", bad_file)
    bad_file.unlink()

print("Image cleaning and conversion completed.")