
import gc
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor, to_pil_image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "bryandlee/animegan2-pytorch:main"
MODEL_STYLE = "paprika"
MAX_SIZE = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None


# ============================================================
# ANIME MODEL — LOAD ONLY WHEN NEEDED
# ============================================================

def get_model():
    global MODEL

    if MODEL is None:
        print("Loading AnimeGANv2...")
        MODEL = torch.hub.load(
            MODEL_REPO,
            "generator",
            pretrained=MODEL_STYLE,
            trust_repo=True,
        )
        MODEL = MODEL.to(DEVICE).eval()
        print("AnimeGANv2 ready.")

    return MODEL


# ============================================================
# COMMON IMAGE PREPARATION
# ============================================================

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size

    working = image.copy()
    working.thumbnail(
        (MAX_SIZE, MAX_SIZE),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(working)
    height, width = array.shape[:2]

    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32

    if pad_h or pad_w:
        array = cv2.copyMakeBorder(
            array,
            0, pad_h, 0, pad_w,
            cv2.BORDER_REFLECT_101,
        )

    tensor = to_tensor(Image.fromarray(array)).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0

    return tensor, width, height, original_size, image


def fit_output(result, width, height, original_size):
    result = result.crop((0, 0, width, height))

    if result.size != original_size:
        result = result.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    return result


# ============================================================
# STYLE 1 — ANIME
# ============================================================

def make_anime(image):
    model = get_model()

    tensor, width, height, original_size, original = prepare_image(
        image
    )

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        prediction = model(tensor)

    prediction = prediction[0].detach().cpu()
    prediction = (prediction * 0.5 + 0.5).clamp(0, 1)
    result = to_pil_image(prediction)

    result = fit_output(
        result,
        width,
        height,
        original_size,
    )

    return original, result


# ============================================================
# STYLE 2 — POP ART
# ============================================================

def make_pop_art(image):
    """
    Free local Pop Art effect:
    smooth colors, posterized color regions, bold outlines.
    """

    original = ImageOps.exif_transpose(image).convert("RGB")
    original_size = original.size

    working = original.copy()
    working.thumbnail(
        (MAX_SIZE, MAX_SIZE),
        Image.Resampling.LANCZOS,
    )

    rgb = np.asarray(working)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Reduce fine photographic texture.
    smoothed = cv2.bilateralFilter(
        bgr,
        d=9,
        sigmaColor=100,
        sigmaSpace=100,
    )

    # Posterize colors into bold, discrete color regions.
    step = 48
    posterized = (
        (smoothed.astype(np.int16) // step) * step
        + step // 2
    )
    posterized = np.clip(
        posterized,
        0,
        255,
    ).astype(np.uint8)

    # Detect major contours.
    gray = cv2.cvtColor(
        smoothed,
        cv2.COLOR_BGR2GRAY,
    )
    edges = cv2.Canny(
        gray,
        threshold1=70,
        threshold2=150,
    )

    # Make the outlines slightly stronger.
    kernel = np.ones((2, 2), dtype=np.uint8)
    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )

    # Apply dark graphic outlines.
    posterized[edges > 0] = (25, 20, 35)

    result_rgb = cv2.cvtColor(
        posterized,
        cv2.COLOR_BGR2RGB,
    )

    result = Image.fromarray(result_rgb)

    if result.size != original_size:
        result = result.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    return original, result


# ============================================================
# MAIN TRANSFORMATION FUNCTION
# ============================================================

def transform_image(input_image, selected_style):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    try:
        if selected_style == "Anime":
            original, result = make_anime(input_image)
            suffix = "anime"

        elif selected_style == "Pop Art":
            original, result = make_pop_art(input_image)
            suffix = "pop_art"

        else:
            raise gr.Error(
                "This style is not implemented yet."
            )

        output_path = OUTPUT_DIR / f"latest_{suffix}.png"
        result.save(output_path, format="PNG")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            original,
            result,
            str(output_path),
            f"{selected_style} transformation complete.",
        )

    except gr.Error:
        raise

    except Exception as error:
        raise gr.Error(
            f"Transformation failed: {error}"
        ) from error


# ============================================================
# USER INTERFACE
# ============================================================

CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
}
#app-header {
    text-align: center;
    padding: 20px 8px 12px;
}
#app-header h1 {
    font-size: 2.2rem;
    margin-bottom: 6px;
}
#convert-button {
    min-height: 48px;
    font-weight: 600;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="AI Art Studio",
) as demo:

    gr.HTML(
        """
        <div id="app-header">
            <h1>AI Art Studio</h1>
            <p>Transform your photographs with free local art styles.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload your photograph",
                type="pil",
                sources=["upload"],
            )

            style_selector = gr.Dropdown(
                choices=["Anime", "Pop Art"],
                value="Anime",
                label="Choose an art style",
            )

            convert_button = gr.Button(
                "Transform Image",
                variant="primary",
                elem_id="convert-button",
            )

        with gr.Column():
            original_preview = gr.Image(
                label="Original",
                type="pil",
            )

            result_preview = gr.Image(
                label="Transformed Result",
                type="pil",
            )

    download_file = gr.File(
        label="Download your image",
        interactive=False,
    )

    status = gr.Textbox(
        label="Status",
        value="Upload an image and choose a style.",
        interactive=False,
    )

    convert_button.click(
        fn=transform_image,
        inputs=[input_image, style_selector],
        outputs=[
            original_preview,
            result_preview,
            download_file,
            status,
        ],
    )

    gr.Markdown(
        """
        **Available now:** Anime and Pop Art.

        **Planned:** Surrealism, Art Nouveau, Impressionism and Art Deco.
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        server_name="127.0.0.1",
        inbrowser=True,
        show_error=True,
    )
