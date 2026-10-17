
from pathlib import Path
import gc

import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor, to_pil_image

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Free, locally running AnimeGANv2 model
MODEL_REPO = "bryandlee/animegan2-pytorch:main"
MODEL_NAME = "paprika"

MAX_SIDE = 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ==================================================
# FIND INPUT PHOTOS
# ==================================================

def find_input_images():
    """Find source photos beside this script."""

    images = []

    for path in PROJECT_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Avoid processing previous outputs as new inputs.
        if path.stem.lower().endswith(("_anime", "_animegan", "_edges")):
            continue

        images.append(path)

    return sorted(images, key=lambda p: p.name.lower())


# ==================================================
# PREPARE IMAGE
# ==================================================

def prepare_image(image_path):
    """Load the photo, resize proportionally, and pad dimensions."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    original_size = image.size

    # Reduce very large inputs without changing the aspect ratio.
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    array = np.asarray(image)
    height, width = array.shape[:2]

    # Pad to multiples of 32 rather than stretching the image.
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32

    if pad_h or pad_w:
        array = cv2.copyMakeBorder(
            array,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT_101,
        )

    padded_image = Image.fromarray(array)

    # AnimeGANv2 expects image values approximately in [-1, 1].
    tensor = to_tensor(padded_image).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0

    return tensor, width, height, original_size


# ==================================================
# GENERATE ANIME IMAGE
# ==================================================

def generate_image(model, device, input_path):
    output_path = OUTPUT_DIR / f"{input_path.stem}_animegan.png"

    try:
        tensor, width, height, original_size = prepare_image(input_path)
        tensor = tensor.to(device)

        print(f"\nProcessing: {input_path.name}")

        with torch.inference_mode():
            prediction = model(tensor)

        # Convert model output from [-1, 1] to an RGB image.
        prediction = prediction[0].detach().cpu()
        prediction = (prediction * 0.5 + 0.5).clamp(0, 1)
        result = to_pil_image(prediction)

        # Remove any padding introduced for model compatibility.
        result = result.crop((0, 0, width, height))

        # Restore the original dimensions if the input was resized.
        if result.size != original_size:
            result = result.resize(
                original_size,
                Image.Resampling.LANCZOS,
            )

        result.save(output_path, format="PNG")

        print("Saved:", output_path.resolve())
        return True

    except Exception as error:
        print(f"FAILED: {input_path.name}")
        print("Reason:", error)
        return False

    finally:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    inputs = find_input_images()

    if not inputs:
        raise FileNotFoundError(
            "No source photos found beside test_style_transfer.py."
        )

    print("Images found:", len(inputs))

    # The code and pretrained weights download the first time.
    print("Loading the free AnimeGANv2 model...")

    model = torch.hub.load(
        MODEL_REPO,
        "generator",
        pretrained=MODEL_NAME,
        trust_repo=True,
    )

    model = model.to(device).eval()

    successful = 0
    failed = 0

    try:
        for input_path in inputs:
            if generate_image(model, device, input_path):
                successful += 1
            else:
                failed += 1
    finally:
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nProcessing complete.")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Output folder:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
