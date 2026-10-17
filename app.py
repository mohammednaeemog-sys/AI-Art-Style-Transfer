import gc
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms.functional import to_tensor, to_pil_image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
EXAMPLES_DIR = PROJECT_DIR / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

ANIME_REPO = "bryandlee/animegan2-pytorch:main"
ANIME_CHECKPOINT = "paprika"
MAX_SIZE = 1024
NST_MAX_SIDE = 384
NST_STEPS = 160

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANIME_MODEL = None
VGG_FEATURES = None
FAST_STYLE_MODELS = {}
STARRY_NIGHT_MODEL = None
DIFFUSION_PIPELINE = None
DIFFUSION_PIPELINE_KEY = None
ACTIVE_LORA_REPO = None
LORA_FAILURES = {}
DIFFUSION_PIPELINE_OFFLOADED_KEY = None
LAST_RENDER_NOTE = ""
DIFFUSION_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_canny"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CONTROLNET_ID = "diffusers/controlnet-canny-sdxl-1.0"
CYBERPUNK_MODEL_ID = (
    "AdamOswald1/Cyberpunk-Anime-Diffusion_with_support_for_Gen-Imp_characters"
)
MODEL_CACHE_DIR = PROJECT_DIR / "models" / "fast_neural_style"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LAST_RESULT_PATH = None
LAST_STYLE = None

STYLES = [
    "Anime",
    "Starry Night (Neural)",
    "Candy (Neural)",
    "Mosaic (Neural)",
    "Rain Princess (Neural)",
    "Udnie (Neural)",
    "Pop Art",
    "Art Nouveau",
    "Ukiyo-e",
    "Impressionism",
    "Cyberpunk",
    "Surrealism",
    "Art Deco",
    "Watercolor",
    "Pencil Sketch",
    "Oil Painting",
    "Pixel Art",
    "Reference Style Transfer",
]

STYLE_DESCRIPTIONS = {
    "Anime": "AnimeGANv2 pretrained neural model.",
    "Starry Night (Neural)": "A dedicated trained feed-forward model for Van Gogh-inspired Starry Night colors and brush texture.",
    "Candy (Neural)": "A pretrained neural style-transfer model with a vivid, painterly Candy style.",
    "Mosaic (Neural)": "A pretrained neural style-transfer model with a mosaic/tile-like artistic style.",
    "Rain Princess (Neural)": "A pretrained neural style-transfer model with a colorful Rain Princess painting style.",
    "Udnie (Neural)": "A pretrained neural style-transfer model based on the Udnie artwork.",
    "Pop Art": "Stable Diffusion + Canny ControlNet for comic-book colors, outlines and halftone aesthetics.",
    "Art Nouveau": "Diffusion-generated ornamental contours, botanical curves and vintage poster tones.",
    "Ukiyo-e": "Diffusion-generated Japanese woodblock-print-inspired illustration.",
    "Impressionism": "Diffusion-generated painterly brushstrokes and expressive color.",
    "Cyberpunk": "Diffusion-generated neon cyberpunk concept art with cyan/magenta lighting.",
    "Surrealism": "Diffusion-generated dreamlike fine-art stylization.",
    "Art Deco": "Diffusion-generated geometric Art Deco poster illustration.",
    "Watercolor": "Diffusion-generated watercolor illustration and pigment washes.",
    "Pencil Sketch": "Diffusion-generated graphite drawing and crosshatching.",
    "Oil Painting": "Diffusion-generated oil-paint surface and visible brushstrokes.",
    "Pixel Art": "Diffusion-generated pixel-art illustration with clear pixel clusters.",
    "Reference Style Transfer": "Slow experimental VGG19 transfer using the artwork you upload. It transfers texture/color and may produce less coherent results than dedicated style models.",
}


# ============================================================
# HIGH-QUALITY DIFFUSION STYLE PROMPTS
# These styles now use Stable Diffusion img2img + Canny ControlNet,
# rather than only the previous OpenCV filters.
# ============================================================

DIFFUSION_STYLE_PROMPTS = {
    "Anime": (
        "Japanese anime illustration, polished cel shading, clean linework, appealing anime color design, "
        "detailed animated background painting; preserve the photographed subject's silhouette, pose, "
        "number of people or animals, and broad composition"
    ),
    "Pop Art": (
        "bold Pop Art comic-book screenprint illustration, flat saturated primary colors, "
        "strong crisp black ink outlines, Ben-Day halftone dots, graphic poster composition, "
        "simplified color shapes, clean high-contrast illustration, professional comic art; "
        "stylize the supplied photo while retaining the main subject, pose, number of visible "
        "subjects, camera angle and overall composition"
    ),
    "Art Nouveau": (
        "Art Nouveau decorative illustration, elegant flowing organic contours, ornamental "
        "botanical curves and floral motifs, muted teal, sage green, antique gold and warm cream, "
        "hand-inked linework, vintage lithograph finish, sophisticated decorative poster; "
        "keep the photograph's main subject, pose and broad composition recognizable"
    ),
    "Ukiyo-e": (
        "traditional Japanese ukiyo-e woodblock print, flat areas of indigo blue, warm ivory, "
        "muted vermilion and dusty teal, elegant carved ink outlines, simplified planes, visible "
        "woodblock print texture, classic Japanese printmaking aesthetics; retain the source "
        "photo's main subject silhouette, pose and spatial layout"
    ),
    "Impressionism": (
        "high-quality Impressionist oil painting, visible short broken brushstrokes, vibrant "
        "natural color, luminous atmospheric light, soft edges in the background, expressive "
        "painted surfaces, richly layered painterly texture, museum-quality fine art; preserve "
        "the main subject and overall arrangement of the supplied photograph"
    ),
    "Cyberpunk": (
        "high-quality cyberpunk concept-art illustration, dramatic electric cyan and magenta neon, "
        "deep midnight shadows, vivid rim lighting, luminous signs and subtle reflected neon, "
        "cinematic futuristic atmosphere, glossy highlights, crisp detailed digital painting; "
        "transform the visual style of the supplied photo while keeping the main subject and "
        "its silhouette, pose, and approximate position"
    ),
    "Surrealism": (
        "surrealist fine-art painting, dreamlike but coherent visual atmosphere, unexpected yet "
        "harmonious color relationships, imaginative lighting, painterly surfaces, unusual scale "
        "and poetic mood, detailed gallery-quality artwork; preserve the main subject and its "
        "recognizable silhouette from the supplied image"
    ),
    "Art Deco": (
        "elegant Art Deco poster illustration, geometric composition, streamlined shapes, "
        "symmetrical ornamental accents, deep navy, antique gold, jade teal and ivory, clean "
        "contours, sophisticated vintage screenprint, premium decorative graphic art; retain the "
        "photo's principal subject, pose and broad layout"
    ),
    "Watercolor": (
        "traditional watercolor illustration on textured cotton paper, transparent pigment washes, "
        "soft color blooms, delicate edges, controlled wet-on-wet textures, luminous highlights, "
        "subtle paper grain, beautiful hand-painted result; retain the source photo's main subject "
        "and its position"
    ),
    "Pencil Sketch": (
        "detailed graphite pencil drawing on white textured paper, delicate crosshatching, "
        "confident contour lines, nuanced graphite shading, hand-drawn strokes, realistic sketchbook "
        "study, monochrome only; preserve the main subject, pose and arrangement"
    ),
    "Oil Painting": (
        "rich traditional oil painting, visible confident brushstrokes, layered pigment, subtle "
        "impasto, rich controlled colors, painterly edges, fine-art canvas texture, carefully "
        "rendered subject; preserve the source photograph's main subject and composition"
    ),
    "Pixel Art": (
        "high-quality colorful retro pixel-art illustration, clearly visible square pixel clusters, "
        "restricted but vibrant game palette, readable subject silhouette, polished 16-bit game-art "
        "aesthetic, crisp nearest-neighbor-like pixel edges, large deliberate pixel blocks, not a "
        "photograph or pencil drawing; preserve the main subject and approximate layout"
    ),
    "Starry Night (Neural)": (
        "Van Gogh inspired Starry Night oil painting, swirling cobalt and ultramarine brush strokes, "
        "bright golden stars, expressive thick painted texture, retain the source subject's silhouette "
        "and the broad composition"
    ),
    "Candy (Neural)": (
        "vivid painterly style transfer, bright colorful swirling paint textures, dynamic brush marks, "
        "bold abstract color planes, recognizable subject and composition"
    ),
    "Mosaic (Neural)": (
        "mosaic tile painting, small colorful tessellated marks, fragmented color planes, tactile tile "
        "texture, clear subject silhouette and recognizable composition"
    ),
    "Rain Princess (Neural)": (
        "dreamlike painterly illustration inspired by Rain Princess, vivid blues and warm golden tones, "
        "soft textured paint, expressive fine-art surface, preserve the main subject and layout"
    ),
    "Udnie (Neural)": (
        "abstract modernist painting inspired by Udnie, dynamic curved shapes, saturated color planes, "
        "expressive geometric brushwork while retaining the source subject's broad arrangement"
    ),
}

DIFFUSION_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, noisy, corrupted image, random colored speckles, "
    "melted shapes, malformed anatomy, extra limbs, duplicated subject, missing main subject, "
    "unrecognizable subject, messy edges, photorealistic camera texture, jpeg artifacts, watermark, "
    "caption, text overlay, frame, border"
)

# Style-specific model routing. LoRAs are loaded only when that style is used.
# Model cards were checked for base-model compatibility and trigger/weight filenames.
# SD15 LoRAs must not be loaded into the SDXL model; Art Nouveau's Mucha adapter is SDXL.
# Sources: Hugging Face model cards for DesiAEye Pop Art, Mucha, Hiroshige, Monet,
# DAVEinside Vintage Art, Jehugging Oil Painting, and PixelArtRedmond.
DIFFUSION_STYLE_CONFIGS = {
    "Pop Art": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": "DesiAEye/DesiAEye-Pop-Art-LORA",
        "lora_weight_name": "DesiAEye-Pop-Art-LORA.safetensors",
        "lora_weight": 0.78,
        "trigger": "POP ART, pop art illustration",
        "strength_bias": 0.79,
        "control": 0.68,
        "guidance": 8.0,
        "control_end": 0.78,
    },
    "Art Nouveau": {
        # This verified Mucha adapter is SDXL-only, so it uses an SDXL pipeline.
        "family": "sdxl",
        "base_model": SDXL_MODEL_ID,
        "controlnet": SDXL_CONTROLNET_ID,
        "lora_repo": "KappaNeuro/alphonse-mucha-style",
        "lora_weight_name": "Alphonse Mucha Style.safetensors",
        "lora_weight": 0.72,
        "trigger": "Alphonse Mucha Style",
        "strength_bias": 0.76,
        "control": 0.66,
        "guidance": 7.0,
        "control_end": 0.82,
    },
    "Ukiyo-e": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": "espressosession/latent-studio-hiroshige-lora",
        "lora_weight_name": "pytorch_lora_weights.safetensors",
        "lora_weight": 0.78,
        "trigger": "sks, Hiroshige Ukiyo-e woodblock print",
        "strength_bias": 0.73,
        "control": 0.74,
        "guidance": 7.5,
        "control_end": 0.82,
    },
    "Impressionism": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": "J-YOON/lora-monet-sd1.5",
        "lora_weight_name": "pytorch_lora_weights.safetensors",
        "lora_weight": 0.62,
        "trigger": "msl monet",
        "strength_bias": 0.70,
        "control": 0.6,
        "guidance": 7.3,
        "control_end": 0.80,
    },
    "Cyberpunk": {
        # Fine-tuned SD 1.5 model; if the repository cannot be loaded with ControlNet,
        # make_diffusion_style safely falls back to the standard SD 1.5 base + style prompt.
        "family": "sd15",
        "base_model": CYBERPUNK_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": None,
        "lora_weight": 1.0,
        "trigger": "dgs illustration style",
        "strength_bias": 0.80,
        "control": 0.57,
        "guidance": 8.2,
        "control_end": 0.72,
    },
    "Surrealism": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": None,
        "lora_weight": 1.0,
        "trigger": "surrealist fine art painting",
        "strength_bias": 0.80,
        "control": 0.45,
        "guidance": 8.0,
        "control_end": 0.65,
    },
    "Art Deco": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": None,
        "lora_weight": 1.0,
        "trigger": "vintage art deco travel poster",
        "strength_bias": 0.75,
        "control": 0.66,
        "guidance": 7.8,
        "control_end": 0.80,
    },
    "Watercolor": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": None,
        "lora_weight": 1.0,
        "trigger": "traditional watercolor illustration",
        "strength_bias": 0.72,
        "control": 0.66,
        "guidance": 7.2,
        "control_end": 0.82,
    },
    "Pencil Sketch": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": None,
        "lora_weight": 1.0,
        "trigger": "detailed graphite pencil drawing",
        "strength_bias": 0.70,
        "control": 0.82,
        "guidance": 7.0,
        "control_end": 0.88,
    },
    "Oil Painting": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": "Jehugging/oilpaint_lora",
        "lora_weight": 0.72,
        "lora_weight_name": "S_oilpainting-07.safetensors",
        "trigger": "oil painting, light color, pastel, brush touch style, warm",
        "strength_bias": 0.76,
        "control": 0.6,
        "guidance": 7.5,
        "control_end": 0.78,
    },
    "Pixel Art": {
        "family": "sd15",
        "base_model": DIFFUSION_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "lora_repo": "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        "lora_weight_name": "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
        "lora_weight": 0.82,
        "trigger": "PixArFK, pixel art",
        "strength_bias": 0.85,
        "control": 0.45,
        "guidance": 8.0,
        "control_end": 0.68,
    },
}

DIFFUSION_NEGATIVE_PROMPT = (
    "photograph, photorealistic, camera noise, random colored speckles, "
    "glitch, melted shapes, malformed anatomy, extra limbs, duplicated main subject, "
    "missing main subject, unreadable text, watermark, caption, frame, border, "
    "low quality, jpeg artifacts, oversharpening"
)

DIFFUSION_DEFAULT_STEPS = 32

# Dedicated fast-style checkpoints can occasionally fail to download or load.
# These prompt-based ControlNet configurations keep every selector usable.
for _fallback_style in (
    "Anime",
    "Starry Night (Neural)",
    "Candy (Neural)",
    "Mosaic (Neural)",
    "Rain Princess (Neural)",
    "Udnie (Neural)",
):
    DIFFUSION_STYLE_CONFIGS.setdefault(
        _fallback_style,
        {
            "family": "sd15",
            "base_model": DIFFUSION_MODEL_ID,
            "controlnet": CONTROLNET_MODEL_ID,
            "lora_repo": None,
            "lora_weight": 1.0,
            "trigger": "",
            "strength_bias": 0.76,
            "control": 0.66,
            "guidance": 7.5,
            "control_end": 0.80,
        },
    )
DIFFUSION_MAX_SIDE = 512
SDXL_MAX_SIDE = 512


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_rgb(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def get_working_image(image, max_size=MAX_SIZE):
    original = load_rgb(image)
    working = original.copy()
    working.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return original, working


def restore_size(result, original):
    if result.size != original.size:
        result = result.resize(original.size, Image.Resampling.LANCZOS)
    return result


def rgb_to_bgr(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# ============================================================
# ANIME — ANIMEGANV2
# ============================================================

def get_anime_model():
    global ANIME_MODEL
    if ANIME_MODEL is None:
        print("Loading AnimeGANv2...")
        ANIME_MODEL = torch.hub.load(
            ANIME_REPO,
            "generator",
            pretrained=ANIME_CHECKPOINT,
            trust_repo=True,
        ).eval()
        print("AnimeGANv2 ready.")
    return ANIME_MODEL.to(DEVICE).eval()


def make_anime(image):
    original, working = get_working_image(image)
    array = np.asarray(working)
    height, width = array.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32
    if pad_h or pad_w:
        array = cv2.copyMakeBorder(
            array, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )

    tensor = to_tensor(Image.fromarray(array)).unsqueeze(0)
    tensor = (tensor * 2 - 1).to(DEVICE)
    with torch.inference_mode():
        prediction = get_anime_model()(tensor)

    prediction = prediction[0].detach().cpu()
    prediction = (prediction * 0.5 + 0.5).clamp(0, 1)
    result = to_pil_image(prediction).crop((0, 0, width, height))
    return restore_size(result, original)


# ============================================================
# POP ART — COLOR BLOCKS, INK, HALFTONE
# ============================================================

def make_pop_art(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    smooth = cv2.bilateralFilter(bgr, 9, 110, 110)
    poster = (smooth // 64) * 64
    poster = np.clip(poster.astype(np.int16) + 22, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 65, 145), np.ones((2, 2), np.uint8))
    poster[edges > 0] = (18, 18, 28)

    h, w = gray.shape
    spacing = max(6, min(12, min(h, w) // 45))
    yy, xx = np.mgrid[0:h, 0:w]
    radius = max(2, spacing // 3)
    dots = ((xx % spacing - spacing // 2) ** 2 +
            (yy % spacing - spacing // 2) ** 2) <= radius ** 2
    mask = dots & (gray > 125) & (edges == 0)
    poster[mask] = (poster[mask].astype(np.float32) * 0.35).astype(np.uint8)
    return restore_size(bgr_to_pil(poster), original)


# ============================================================
# ART NOUVEAU
# ============================================================

def make_art_nouveau(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    result = cv2.stylization(bgr, sigma_s=60, sigma_r=0.32)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 0.88, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    parchment = np.full_like(result, (178, 211, 226))
    result = cv2.addWeighted(result, 0.84, parchment, 0.16, 3)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 85, 175), np.ones((2, 2), np.uint8))
    result[edges > 0] = (35, 62, 45)
    return restore_size(bgr_to_pil(result), original)


# ============================================================
# UKIYO-E
# ============================================================

def make_ukiyoe(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    smooth = cv2.bilateralFilter(bgr, 11, 100, 100)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    palette = np.array([
        [54, 35, 24], [92, 65, 44], [145, 105, 72],
        [185, 157, 119], [218, 205, 174], [235, 229, 211],
    ], dtype=np.uint8)
    result = palette[np.clip(gray // 43, 0, len(palette) - 1)]
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    blue_green = (hsv[:, :, 0] > 75) & (hsv[:, :, 0] < 115)
    result[blue_green] = (
        result[blue_green].astype(np.float32) * 0.55
        + np.array([130, 105, 58], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    edges = cv2.dilate(cv2.Canny(gray, 55, 135), np.ones((2, 2), np.uint8))
    result[edges > 0] = (28, 26, 32)
    return restore_size(bgr_to_pil(result), original)


# ============================================================
# IMPRESSIONISM
# ============================================================

def make_impressionism(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    result = cv2.stylization(bgr, sigma_s=75, sigma_r=0.28)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.22, beta=5)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    small = cv2.resize(result, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    texture = cv2.resize(small, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
    result = cv2.addWeighted(result, 0.78, texture, 0.22, 0)
    return restore_size(bgr_to_pil(result), original)


# ============================================================
# CYBERPUNK — FIXED CHANNEL BROADCAST FOR EDGE GLOW
# ============================================================

def make_cyberpunk(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    smooth = cv2.bilateralFilter(bgr, 9, 90, 90)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    hsv[:, :, 0] = np.where(
        hue < 25, (hue + 145) % 180,
        np.where(hue < 90, (hue + 35) % 180, (hue + 105) % 180)
    )
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.55 + 18, 0, 255)
    hsv[:, :, 2] = np.clip((hsv[:, :, 2] - 100) * 1.35 + 100, 0, 255)
    neon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 75, 165), np.ones((2, 2), np.uint8))
    glow_color = np.zeros_like(neon)
    glow_color[:] = (255, 80, 180)
    glow_mask = cv2.GaussianBlur(edges, (0, 0), sigmaX=3)
    glow = glow_color.astype(np.float32) * (
        glow_mask[:, :, None].astype(np.float32) / 255.0
    )
    neon = cv2.addWeighted(neon, 0.82, np.clip(glow, 0, 255).astype(np.uint8), 0.65, 0)
    neon[edges > 0] = (220, 75, 255)
    return restore_size(bgr_to_pil(neon), original)


# ============================================================
# SURREALISM
# ============================================================

def make_surrealism(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    painted = cv2.stylization(bgr, sigma_s=60, sigma_r=0.45)
    hsv = cv2.cvtColor(painted, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + 12) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return restore_size(
        bgr_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)),
        original,
    )


# ============================================================
# ART DECO
# ============================================================

def make_art_deco(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    smooth = cv2.bilateralFilter(bgr, 9, 90, 90)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    palette = np.array([
        [35, 28, 48], [65, 77, 78], [87, 126, 121],
        [92, 157, 193], [165, 202, 218], [210, 224, 232],
    ], dtype=np.uint8)
    indices = np.zeros_like(gray, dtype=np.uint8)
    for threshold in [35, 75, 115, 155, 195]:
        indices += (gray >= threshold).astype(np.uint8)
    result = palette[indices]
    result[cv2.Canny(gray, 90, 180) > 0] = (28, 25, 38)
    return restore_size(bgr_to_pil(result), original)


# ============================================================
# WATERCOLOR
# ============================================================

def make_watercolor(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    result = cv2.stylization(bgr, sigma_s=80, sigma_r=0.22)
    blur = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
    result = cv2.addWeighted(result, 0.72, blur, 0.28, 10)
    return restore_size(bgr_to_pil(result), original)


# ============================================================
# PENCIL SKETCH
# ============================================================

def make_pencil_sketch(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    sketch, _ = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.06)
    return restore_size(bgr_to_pil(sketch), original)


# ============================================================
# OIL PAINTING-INSPIRED EFFECT
# ============================================================

def make_oil_painting(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    smooth = cv2.bilateralFilter(bgr, 13, 140, 140)
    smooth = cv2.stylization(smooth, sigma_s=45, sigma_r=0.32)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.18, beta=3)
    return restore_size(bgr_to_pil(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)), original)


# ============================================================
# PIXEL ART
# ============================================================

def make_pixel_art(image):
    original, working = get_working_image(image)
    bgr = rgb_to_bgr(working)
    h, w = bgr.shape[:2]
    tiny_w = max(24, w // 16)
    tiny_h = max(24, h // 16)
    tiny = cv2.resize(bgr, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA)
    tiny = (tiny // 64) * 64
    pixel = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    return restore_size(bgr_to_pil(pixel), original)


# ============================================================
# PRETRAINED FAST NEURAL STYLE MODELS
# ============================================================

class FastNeuralTransformer(torch.nn.Module):
    """TransformerNet architecture used by the official PyTorch fast-neural-style example."""

    def __init__(self):
        super().__init__()

        class ConvLayer(torch.nn.Module):
            def __init__(self, in_c, out_c, kernel, stride):
                super().__init__()
                self.reflection_pad = torch.nn.ReflectionPad2d(kernel // 2)
                self.conv2d = torch.nn.Conv2d(in_c, out_c, kernel, stride)

            def forward(self, x):
                return self.conv2d(self.reflection_pad(x))

        class ResidualBlock(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = ConvLayer(channels, channels, 3, 1)
                self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.conv2 = ConvLayer(channels, channels, 3, 1)
                self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = self.relu(self.in1(self.conv1(x)))
                return self.in2(self.conv2(out)) + x

        class UpsampleConvLayer(torch.nn.Module):
            def __init__(self, in_c, out_c, kernel, stride, upsample=None):
                super().__init__()
                self.upsample = upsample
                self.reflection_pad = torch.nn.ReflectionPad2d(kernel // 2)
                self.conv2d = torch.nn.Conv2d(in_c, out_c, kernel, stride)

            def forward(self, x):
                if self.upsample:
                    x = torch.nn.functional.interpolate(
                        x, mode="nearest", scale_factor=self.upsample
                    )
                return self.conv2d(self.reflection_pad(x))

        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        return self.deconv3(y)


FAST_MODEL_NAMES = {
    "Candy (Neural)": "candy.pth",
    "Mosaic (Neural)": "mosaic.pth",
    "Rain Princess (Neural)": "rain_princess.pth",
    "Udnie (Neural)": "udnie.pth",
}
FAST_MODELS_ZIP_URL = "https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"


def _ensure_fast_model_file(filename):
    target = MODEL_CACHE_DIR / filename
    if target.exists() and target.stat().st_size > 100_000:
        return target

    archive = MODEL_CACHE_DIR / "saved_models.zip"
    if not archive.exists() or archive.stat().st_size < 100_000:
        print("Downloading pretrained fast-neural-style models...")
        try:
            urllib.request.urlretrieve(FAST_MODELS_ZIP_URL, archive)
        except Exception as error:
            raise RuntimeError(
                "Could not download the pretrained style models. Check internet access "
                "and try again. The models are sourced from the PyTorch examples project."
            ) from error

    if not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "The style-model download was not a valid ZIP archive. Retry when internet is available."
        )

    # Extract only the exact required checkpoint basename, never arbitrary archive paths.
    with zipfile.ZipFile(archive) as zf:
        member = next(
            (entry for entry in zf.namelist()
             if Path(entry).name.lower() == filename.lower()),
            None,
        )
        if member is None:
            raise RuntimeError(f"The model archive did not contain {filename}.")
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    if target.stat().st_size < 100_000:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded checkpoint {filename} looks incomplete.")
    return target


def get_fast_style_model(style_label):
    if style_label not in FAST_MODEL_NAMES:
        raise ValueError(f"No pretrained fast-neural model for {style_label}")

    if style_label not in FAST_STYLE_MODELS:
        path = _ensure_fast_model_file(FAST_MODEL_NAMES[style_label])
        model = FastNeuralTransformer()
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Accommodate old InstanceNorm checkpoints containing obsolete running stats.
        for key in list(state.keys()):
            if re.search(r"in\d+\.running_(mean|var)$", key):
                state.pop(key)

        model.load_state_dict(state, strict=True)
        FAST_STYLE_MODELS[style_label] = model.eval()
    return FAST_STYLE_MODELS[style_label].to(DEVICE).eval()


def make_fast_neural_style(image, style_label, blend=0.72):
    """Apply a pretrained fixed-style transformer and blend with the source."""
    original, working = get_working_image(image)
    array = np.asarray(working)
    height, width = array.shape[:2]

    # The network downsamples twice; pad to a multiple of four.
    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4
    if pad_h or pad_w:
        array = cv2.copyMakeBorder(
            array, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )

    tensor = (to_tensor(Image.fromarray(array)).unsqueeze(0) * 255.0).to(DEVICE)
    model = get_fast_style_model(style_label)
    with torch.inference_mode():
        output = model(tensor)[0].detach().cpu()

    output = output.clamp(0, 255).div(255.0)
    styled = to_pil_image(output).crop((0, 0, width, height))
    styled = restore_size(styled, original)
    return Image.blend(original, styled, float(np.clip(blend, 0.0, 1.0)))


# ============================================================
# STARry NIGHT — DEDICATED PRETRAINED MODEL
# ============================================================

class StarryNightNet(torch.nn.Module):
    """Feed-forward Starry Night model documented by Rohanify/Brawnz-StyleTransferSN."""

    def __init__(self):
        super().__init__()

        def conv_bn_relu(in_c, out_c, kernel, stride=1, padding=0):
            return torch.nn.Sequential(
                torch.nn.ReflectionPad2d(padding),
                torch.nn.Conv2d(in_c, out_c, kernel, stride),
                torch.nn.InstanceNorm2d(out_c),
                torch.nn.ReLU(inplace=True),
            )

        class ResBlock(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.block = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(channels, channels, 3),
                    torch.nn.InstanceNorm2d(channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(channels, channels, 3),
                    torch.nn.InstanceNorm2d(channels),
                )

            def forward(self, x):
                return x + self.block(x)

        self.net = torch.nn.Sequential(
            conv_bn_relu(3, 32, 9, padding=4),
            conv_bn_relu(32, 64, 3, stride=2, padding=1),
            conv_bn_relu(64, 128, 3, stride=2, padding=1),
            ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            conv_bn_relu(128, 64, 3, padding=1),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            conv_bn_relu(64, 32, 3, padding=1),
            torch.nn.ReflectionPad2d(4),
            torch.nn.Conv2d(32, 3, 9),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def get_starry_night_model():
    global STARRY_NIGHT_MODEL
    if STARRY_NIGHT_MODEL is None:
        print("Downloading/loading the Starry Night neural model...")
        try:
            weights_path = hf_hub_download(
                repo_id="Rohanify/Brawnz-StyleTransferSN",
                filename="pytorch_model.bin",
            )
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model = StarryNightNet()
            model.load_state_dict(state, strict=True)
            STARRY_NIGHT_MODEL = model.to(DEVICE).eval()
        except Exception as error:
            raise RuntimeError(
                "Could not load the Starry Night neural model. Check internet access "
                "and try again; the model weights are cached after the first download."
            ) from error
    return STARRY_NIGHT_MODEL.to(DEVICE).eval()


def make_starry_night_neural(image, blend=0.72):
    original, working = get_working_image(image)
    array = np.asarray(working)
    height, width = array.shape[:2]
    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4
    if pad_h or pad_w:
        array = cv2.copyMakeBorder(
            array, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )

    tensor = to_tensor(Image.fromarray(array)).unsqueeze(0).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.inference_mode():
        output = get_starry_night_model()(tensor)[0].detach().cpu()

    # The model's final Tanh produces values in [-1, 1].
    output = (output * 0.5 + 0.5).clamp(0, 1)
    styled = to_pil_image(output).crop((0, 0, width, height))
    styled = restore_size(styled, original)
    return Image.blend(original, styled, float(np.clip(blend, 0.0, 1.0)))


# ============================================================
# REFERENCE STYLE TRANSFER USING VGG19
# ============================================================

def get_vgg_features():
    global VGG_FEATURES
    if VGG_FEATURES is None:
        print("Downloading/loading VGG19 weights for neural style transfer...")
        net = vgg19(weights=VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        VGG_FEATURES = net
    return VGG_FEATURES.to(DEVICE).eval()


def gram_matrix(features):
    batch, channels, height, width = features.shape
    values = features.reshape(batch, channels, height * width)
    return torch.bmm(values, values.transpose(1, 2)) / (channels * height * width)


def get_features(net, image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    x = (image_tensor - mean) / std
    wanted = {1, 6, 11, 20, 21, 29}
    activations = {}
    for index, layer in enumerate(net):
        x = layer(x)
        if index in wanted:
            activations[index] = x
        if index >= 29:
            break
    return activations


def make_reference_style_transfer(content_image, style_image, strength):
    if style_image is None:
        raise gr.Error(
            "Upload a reference artwork in the Style Reference field "
            "when using Reference Style Transfer."
        )

    content_original, content = get_working_image(content_image, NST_MAX_SIDE)
    style_original, style = get_working_image(style_image, NST_MAX_SIDE)

    content_tensor = to_tensor(content).unsqueeze(0).to(DEVICE)
    style_tensor = to_tensor(style).unsqueeze(0).to(DEVICE)
    net = get_vgg_features()

    with torch.no_grad():
        content_acts = get_features(net, content_tensor)
        style_acts = get_features(net, style_tensor)
        target_content = content_acts[21].detach()
        target_grams = {
            layer: gram_matrix(style_acts[layer]).detach()
            for layer in (1, 6, 11, 20, 29)
        }

    generated = content_tensor.clone().detach().requires_grad_(True)
    # Lower learning rate and more iterations make the optimization less
    # unstable and allow the artwork's broader color/texture patterns to emerge.
    optimizer = torch.optim.Adam([generated], lr=0.012)
    # This is deliberately much stronger than the original setting, which
    # produced only a weak texture overlay at typical slider values.
    style_weight = 300000.0 + float(strength) * 2000000.0

    for _ in range(NST_STEPS):
        optimizer.zero_grad(set_to_none=True)
        generated_acts = get_features(net, generated)
        content_loss = F.mse_loss(generated_acts[21], target_content)

        style_loss = torch.zeros((), device=DEVICE)
        for layer in (1, 6, 11, 20, 29):
            style_loss = style_loss + F.mse_loss(
                gram_matrix(generated_acts[layer]), target_grams[layer]
            )

        # Keep the output spatially smooth and content-like.
        tv_h = torch.mean(torch.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(generated[:, :, :, 1:] - generated[:, :, :, :-1]))
        loss = content_loss + style_weight * style_loss + 0.06 * (tv_h + tv_w)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(0, 1)

    result = to_pil_image(generated.detach().cpu().squeeze(0).clamp(0, 1))
    return restore_size(result, content_original)


# ============================================================
# CONTROLNET IMG2IMG PIPELINE FOR MORE ACCURATE NON-NEURAL PRESETS
# ============================================================

def move_cached_neural_models_to_cpu():
    """Free VRAM before running the larger Stable Diffusion + ControlNet pipeline."""
    global ANIME_MODEL, VGG_FEATURES, STARRY_NIGHT_MODEL

    cached = [ANIME_MODEL, VGG_FEATURES, STARRY_NIGHT_MODEL]
    cached.extend(FAST_STYLE_MODELS.values())

    for model in cached:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_diffusion_pipeline(style_config):
    """Load a fresh compatible pipeline and keep it on CPU until optional LoRA is attached.

    Including the adapter in the cache key avoids swapping LoRAs after offload hooks have
    been installed. Hub downloads are cached locally, so model files are not re-downloaded
    every time the app is restarted.
    """
    global DIFFUSION_PIPELINE, DIFFUSION_PIPELINE_KEY
    global DIFFUSION_PIPELINE_OFFLOADED_KEY, ACTIVE_LORA_REPO

    family = style_config["family"]
    base_model = style_config["base_model"]
    controlnet_id = style_config["controlnet"]
    pipeline_key = (
        f"{family}|{base_model}|{controlnet_id}|"
        f"{style_config.get('lora_repo')}|{style_config.get('lora_weight_name')}"
    )

    if DIFFUSION_PIPELINE is not None and DIFFUSION_PIPELINE_KEY == pipeline_key:
        return DIFFUSION_PIPELINE

    # Release the previous model before loading a different base / adapter.
    if DIFFUSION_PIPELINE is not None:
        DIFFUSION_PIPELINE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    DIFFUSION_PIPELINE_KEY = None
    DIFFUSION_PIPELINE_OFFLOADED_KEY = None
    ACTIVE_LORA_REPO = None

    print(f"Loading {family.upper()} + Canny ControlNet from {base_model}...")
    try:
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
    except Exception as error:
        raise RuntimeError(
            "Diffusers could not be imported. Run `python -m pip install -r requirements.txt`. "
            f"Underlying error: {type(error).__name__}: {error}"
        ) from error

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipeline_class = (
        StableDiffusionXLControlNetImg2ImgPipeline
        if family == "sdxl"
        else StableDiffusionControlNetImg2ImgPipeline
    )

    pipe = pipeline_class.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.enable_attention_slicing()
    # Optional VAE memory helpers are present in most supported Diffusers versions.
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # Do NOT call enable_model_cpu_offload before load_lora_weights.
    # Attaching adapters to a pipeline with offload hooks can fail on some versions.
    DIFFUSION_PIPELINE = pipe
    DIFFUSION_PIPELINE_KEY = pipeline_key
    print("Pipeline loaded. Attaching style adapter before GPU offload...")
    return DIFFUSION_PIPELINE


def _set_style_lora(pipe, style_config):
    """Load a compatible LoRA; on failure, visibly continue with prompt + ControlNet."""
    global ACTIVE_LORA_REPO, LORA_FAILURES

    repository = style_config.get("lora_repo")
    if not repository:
        return "Using style-specific prompt + Canny ControlNet (no adapter configured)."

    weight_name = style_config.get("lora_weight_name")
    if ACTIVE_LORA_REPO == repository:
        # Pipeline cache key includes adapter repo and file, so this is the same
        # loaded adapter with the same weight. Avoid trying to reload after offload hooks.
        return f"Reusing cached adapter: {repository}."

    if repository in LORA_FAILURES:
        return (
            f"WARNING: adapter `{repository}` failed earlier this session "
            f"({LORA_FAILURES[repository]}). Using the style prompt + ControlNet."
        )

    try:
        kwargs = {"adapter_name": "art_style"}
        if weight_name:
            kwargs["weight_name"] = weight_name
        pipe.load_lora_weights(repository, **kwargs)
        pipe.set_adapters(
            ["art_style"],
            adapter_weights=[float(style_config.get("lora_weight", 0.75))],
        )
        ACTIVE_LORA_REPO = repository
        return f"Loaded pretrained style adapter: {repository}."
    except Exception as error:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        ACTIVE_LORA_REPO = None
        detail = f"{type(error).__name__}: {error}"
        LORA_FAILURES[repository] = detail[:500]
        print(f"Optional LoRA load failed for {repository} ({weight_name or 'default file'}): {detail}")
        return (
            f"WARNING: adapter `{repository}` did not load "
            f"({detail[:260]}). This render will use the style-specific prompt + Canny ControlNet. "
            "The full cause is printed in the VS Code terminal."
        )


def enable_pipeline_offload(pipe, style_config):
    """Enable memory-saving device hooks only after model and adapter loading."""
    global DIFFUSION_PIPELINE_OFFLOADED_KEY

    key = DIFFUSION_PIPELINE_KEY
    if DIFFUSION_PIPELINE_OFFLOADED_KEY == key:
        return

    if torch.cuda.is_available():
        # Sequential offload is slower but safest for SDXL on an 8 GB GPU.
        if style_config["family"] == "sdxl":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    DIFFUSION_PIPELINE_OFFLOADED_KEY = key


def prepare_diffusion_images(image, max_side=DIFFUSION_MAX_SIDE):
    """Resize without changing aspect ratio; produce Canny guidance for broad contours."""
    original, working = get_working_image(image, max_size=max_side)
    width = max(64, (working.width // 8) * 8)
    height = max(64, (working.height // 8) * 8)
    working = working.resize((width, height), Image.Resampling.LANCZOS)

    rgb = np.asarray(working)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=75, threshold2=160)
    return original, working, Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def _render_controlnet(pipe, prompt, working, control_image, config, denoise_strength,
                       control_scale, steps, seed_value, guidance, control_end):
    generator = torch.Generator(device="cpu").manual_seed(int(seed_value))
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            negative_prompt=DIFFUSION_NEGATIVE_PROMPT,
            image=working,
            control_image=control_image,
            strength=float(denoise_strength),
            controlnet_conditioning_scale=float(control_scale),
            control_guidance_end=float(control_end),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
        ).images[0]


def make_diffusion_style(
    image,
    style_name,
    style_strength=0.80,
    structure_strength=0.60,
    steps=DIFFUSION_DEFAULT_STEPS,
    seed=-1,
):
    """Run style-specific SD/ControlNet; use the matching LoRA where possible.

    Model failures trigger a controlled SD 1.5 + Canny fallback. Adapter failures are
    already visible in the status message and do not prevent image generation.
    """
    global LAST_RENDER_NOTE

    if style_name not in DIFFUSION_STYLE_PROMPTS or style_name not in DIFFUSION_STYLE_CONFIGS:
        raise ValueError(f"No diffusion configuration found for {style_name!r}")

    config = dict(DIFFUSION_STYLE_CONFIGS[style_name])
    pipeline_note = ""
    max_side = SDXL_MAX_SIDE if config["family"] == "sdxl" else DIFFUSION_MAX_SIDE
    original, working, control_image = prepare_diffusion_images(image, max_side=max_side)

    style_strength = float(np.clip(style_strength, 0.30, 0.95))
    structure_strength = float(np.clip(structure_strength, 0.25, 0.95))
    steps = int(np.clip(steps, 20, 40))
    style_bias = float(config.get("strength_bias", 0.75))
    # For img2img: higher strength -> more creative/style-dominant, lower -> source fidelity.
    denoise_strength = float(np.clip(
        0.32 + 0.56 * style_strength + (style_bias - 0.75) * 0.14,
        0.45, 0.88,
    ))
    control_scale = float(np.clip(
        float(config.get("control", 0.58)) * (structure_strength / 0.60),
        0.15, 1.0,
    ))
    guidance = float(config.get("guidance", 7.5))
    control_end = float(config.get("control_end", 0.80))
    try:
        requested_seed = int(seed) if seed is not None else -1
    except (TypeError, ValueError):
        requested_seed = -1
    if requested_seed < 0:
        seed_value = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed_value = requested_seed

    trigger = config.get("trigger", "")
    prompt = f"{trigger}, {DIFFUSION_STYLE_PROMPTS[style_name]}".strip(", ")

    # Try style-specific base and compatible adapter first.
    try:
        pipe = get_diffusion_pipeline(config)
        lora_note = _set_style_lora(pipe, config)
        enable_pipeline_offload(pipe, config)
    except Exception as primary_model_error:
        # Cyberpunk fine-tune and SDXL may fail on a given machine. Fall back to
        # the known SD 1.5 + Canny pipeline, keeping the same style-specific prompt.
        print(
            f"Preferred pipeline for {style_name} failed: "
            f"{type(primary_model_error).__name__}: {primary_model_error}"
        )
        fallback_config = {
            "family": "sd15",
            "base_model": DIFFUSION_MODEL_ID,
            "controlnet": CONTROLNET_MODEL_ID,
            "lora_repo": None,
            "lora_weight": 1.0,
            "trigger": "",
            "strength_bias": style_bias,
            "control": float(config.get("control", 0.58)),
            "guidance": guidance,
            "control_end": control_end,
        }
        pipeline_note = (
            f"WARNING: preferred base model `{config['base_model']}` failed to load "
            f"({type(primary_model_error).__name__}: {primary_model_error}). "
            "Using Stable Diffusion 1.5 + Canny ControlNet instead. "
        )
        config = fallback_config
        prompt = DIFFUSION_STYLE_PROMPTS[style_name]
        max_side = DIFFUSION_MAX_SIDE
        original, working, control_image = prepare_diffusion_images(image, max_side=max_side)
        pipe = get_diffusion_pipeline(config)
        lora_note = _set_style_lora(pipe, config)
        enable_pipeline_offload(pipe, config)

    LAST_RENDER_NOTE = (
        f"{pipeline_note}{lora_note} Style strength {denoise_strength:.2f}; "
        f"composition guidance {control_scale:.2f}; {steps} steps; seed {seed_value}."
    )

    try:
        output = _render_controlnet(
            pipe, prompt, working, control_image, config, denoise_strength,
            control_scale, steps, seed_value, guidance, control_end
        )
    except torch.cuda.OutOfMemoryError:
        if not torch.cuda.is_available():
            raise
        print("CUDA OOM: retrying at 448px and 24 steps.")
        gc.collect()
        torch.cuda.empty_cache()
        original, working, control_image = prepare_diffusion_images(image, max_side=448)
        output = _render_controlnet(
            pipe, prompt, working, control_image, config,
            min(denoise_strength, 0.78), control_scale,
            24, seed_value, guidance, control_end,
        )
        LAST_RENDER_NOTE += " Automatic memory fallback: 448px and 24 steps."

    except Exception as primary_render_error:
        # A custom fine-tuned checkpoint may load but not be compatible with a ControlNet
        # pipeline. Retry the render using the baseline SD 1.5 + Canny architecture.
        if (
            config.get("base_model") == DIFFUSION_MODEL_ID
            and config.get("family") == "sd15"
            and config.get("controlnet") == CONTROLNET_MODEL_ID
            and not config.get("lora_repo")
        ):
            raise RuntimeError(
                f"Image generation failed ({type(primary_render_error).__name__}: "
                f"{primary_render_error}). See the terminal for the full traceback."
            ) from primary_render_error

        print(
            f"Render with selected base failed: "
            f"{type(primary_render_error).__name__}: {primary_render_error}"
        )
        fallback_config = {
            "family": "sd15",
            "base_model": DIFFUSION_MODEL_ID,
            "controlnet": CONTROLNET_MODEL_ID,
            "lora_repo": None,
            "lora_weight": 1.0,
            "trigger": "",
            "strength_bias": style_bias,
            "control": float(config.get("control", 0.58)),
            "guidance": guidance,
            "control_end": control_end,
        }
        pipe = get_diffusion_pipeline(fallback_config)
        lora_note = _set_style_lora(pipe, fallback_config)
        enable_pipeline_offload(pipe, fallback_config)
        fallback_original, working, control_image = prepare_diffusion_images(image, max_side=448)
        output = _render_controlnet(
            pipe, DIFFUSION_STYLE_PROMPTS[style_name], working, control_image,
            fallback_config, min(denoise_strength, 0.78), control_scale,
            24, seed_value, guidance, control_end,
        )
        original = fallback_original
        LAST_RENDER_NOTE = (
            f"WARNING: the preferred render path failed ({type(primary_render_error).__name__}: "
            f"{primary_render_error}); rendered with baseline SD 1.5 + ControlNet. "
            f"{lora_note} 448px, 24 steps."
        )

    return restore_size(output, original)


# ============================================================
# STYLE ROUTER
# ============================================================

STYLE_FUNCTIONS = {
    "Anime": make_anime,
    "Pop Art": make_pop_art,
    "Art Nouveau": make_art_nouveau,
    "Ukiyo-e": make_ukiyoe,
    "Impressionism": make_impressionism,
    "Cyberpunk": make_cyberpunk,
    "Surrealism": make_surrealism,
    "Art Deco": make_art_deco,
    "Watercolor": make_watercolor,
    "Pencil Sketch": make_pencil_sketch,
    "Oil Painting": make_oil_painting,
    "Pixel Art": make_pixel_art,
    "Starry Night (Neural)": None,
    "Candy (Neural)": None,
    "Mosaic (Neural)": None,
    "Rain Princess (Neural)": None,
    "Udnie (Neural)": None,
    "Reference Style Transfer": None,
}


# ============================================================
# TRANSFORM IMAGE
# ============================================================

def transform_image(
    input_image,
    selected_style,
    style_reference,
    style_strength,
    structure_strength,
    quality_steps,
    seed,
):
    global LAST_RESULT_PATH, LAST_STYLE, LAST_RENDER_NOTE

    if input_image is None:
        raise gr.Error("Please upload a photograph first.")
    if selected_style not in STYLES:
        raise gr.Error("Select one of the listed styles.")

    try:
        original = load_rgb(input_image)
        LAST_RENDER_NOTE = ""

        if selected_style == "Reference Style Transfer":
            result = make_reference_style_transfer(
                original, style_reference, style_strength
            )
            LAST_RENDER_NOTE = (
                "Experimental VGG19 reference transfer: borrows texture/color statistics, "
                "but does not semantically recreate shapes from the style reference."
            )

        elif selected_style == "Anime":
            try:
                result = make_anime(original)
                LAST_RENDER_NOTE = "Used AnimeGANv2."
            except Exception as model_error:
                print(f"AnimeGANv2 failed: {type(model_error).__name__}: {model_error}")
                result = make_diffusion_style(
                    original, "Anime", style_strength, structure_strength, quality_steps, seed
                ) if "Anime" in DIFFUSION_STYLE_CONFIGS else make_diffusion_style(
                    original, "Surrealism", style_strength, structure_strength, quality_steps, seed
                )
                LAST_RENDER_NOTE = (
                    f"WARNING: AnimeGANv2 was unavailable ({type(model_error).__name__}: "
                    f"{model_error}); used a diffusion-based illustration fallback. "
                    f"{LAST_RENDER_NOTE}"
                )

        elif selected_style == "Starry Night (Neural)":
            try:
                result = make_starry_night_neural(original, style_strength)
                LAST_RENDER_NOTE = "Used dedicated Starry Night feed-forward model."
            except Exception as model_error:
                print(f"Starry Night fixed-style model failed: {model_error}")
                result = make_diffusion_style(
                    original, selected_style, style_strength, structure_strength, quality_steps, seed
                )
                LAST_RENDER_NOTE = (
                    f"WARNING: dedicated Starry Night model unavailable ({model_error}); "
                    f"used prompt + ControlNet fallback. {LAST_RENDER_NOTE}"
                )

        elif selected_style in FAST_MODEL_NAMES:
            try:
                result = make_fast_neural_style(original, selected_style, style_strength)
                LAST_RENDER_NOTE = f"Used dedicated {selected_style} checkpoint."
            except Exception as model_error:
                print(f"Fixed-style checkpoint for {selected_style} failed: {model_error}")
                result = make_diffusion_style(
                    original, selected_style, style_strength, structure_strength, quality_steps, seed
                )
                LAST_RENDER_NOTE = (
                    f"WARNING: dedicated {selected_style} model unavailable ({model_error}); "
                    f"used prompt + ControlNet fallback. {LAST_RENDER_NOTE}"
                )

        elif selected_style in DIFFUSION_STYLE_CONFIGS:
            result = make_diffusion_style(
                original,
                selected_style,
                style_strength,
                structure_strength,
                quality_steps,
                seed,
            )

        else:
            function = STYLE_FUNCTIONS.get(selected_style)
            if function is None:
                raise RuntimeError(f"No implementation is configured for style {selected_style!r}.")
            result = function(original)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-z0-9]+", "_", selected_style.lower()).strip("_")
        output_path = OUTPUT_DIR / f"{timestamp}_{slug}.png"
        result.save(output_path, format="PNG")

        LAST_RESULT_PATH = output_path
        LAST_STYLE = selected_style

        return original, result, str(output_path), (
            f"{selected_style} complete. Saved locally: {output_path.name}. {LAST_RENDER_NOTE}"
        )
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(
            f"Transformation failed: {type(error).__name__}: {error}. "
            "Check the VS Code terminal for the full error."
        ) from error
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# APPROVED-ONLY GITHUB PUBLISHING
# ============================================================

def publish_showcase(approved, confirmation_text):
    if not approved:
        raise gr.Error("Tick the approval checkbox first.")
    if confirmation_text.strip() != "PUBLISH":
        raise gr.Error("Type exactly PUBLISH to confirm.")
    if LAST_RESULT_PATH is None or not LAST_RESULT_PATH.is_file():
        raise gr.Error("Generate and review an image before publishing.")

    try:
        inside_repo = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise gr.Error("Configure Git and the origin remote first.") from error

    if inside_repo != "true" or not remote:
        raise gr.Error("This project has no configured GitHub origin.")

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = (LAST_STYLE or "style").lower().replace(" ", "_")
    showcase = EXAMPLES_DIR / f"{timestamp}_{slug}.png"
    relative = showcase.relative_to(PROJECT_DIR).as_posix()

    try:
        with Image.open(LAST_RESULT_PATH) as result:
            result.convert("RGB").save(showcase, "PNG")

        subprocess.run(
            ["git", "add", "--", relative],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        ).stdout.splitlines()

        # Fail closed if any unrelated file was already staged.
        if staged != [relative]:
            subprocess.run(
                ["git", "reset", "--", relative],
                cwd=PROJECT_DIR, capture_output=True, text=True,
            )
            raise gr.Error(
                "Publishing stopped: unrelated files are staged. Unstage them "
                "in VS Code Source Control and retry."
            )

        subprocess.run(
            ["git", "commit", "-m", f"Add approved {LAST_STYLE} showcase"],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
        return f"Published `examples/{showcase.name}`. Only the approved showcase image was staged."
    except gr.Error:
        raise
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise gr.Error(f"GitHub publishing failed: {detail[-1000:]}") from error


# ============================================================
# GRADIO UI
# ============================================================

CSS = """
:root { --studio-ink: #172033; --studio-violet: #635bff; --studio-cyan: #39d6e8; }
.gradio-container {
  max-width: 1280px !important; margin: 0 auto !important; padding: 18px !important;
  background:
    radial-gradient(circle at 4% 0%, rgba(99,91,255,.10), transparent 28rem),
    radial-gradient(circle at 96% 8%, rgba(57,214,232,.10), transparent 30rem),
    #f6f8ff;
  color: var(--studio-ink);
}
#app-header {
  position: relative; overflow: hidden; text-align: left; padding: 32px 34px;
  margin: 4px 0 22px; border-radius: 24px; color: white;
  background: linear-gradient(120deg, #171d42 0%, #4942a8 48%, #267b9c 100%);
  box-shadow: 0 18px 45px rgba(36,42,105,.20);
  animation: studioFadeIn .65s ease both;
}
#app-header:after {
  content: ""; position: absolute; width: 260px; height: 260px; right: -52px; top: -135px;
  border-radius: 50%; background: rgba(255,255,255,.11);
  box-shadow: 0 0 0 34px rgba(255,255,255,.04), 0 0 0 70px rgba(255,255,255,.025);
  animation: studioFloat 8s ease-in-out infinite alternate;
}
#app-header .hero-kicker {
  position: relative; z-index: 1; display: inline-block; padding: 6px 11px;
  margin-bottom: 10px; border: 1px solid rgba(255,255,255,.28);
  border-radius: 999px; background: rgba(255,255,255,.10); font-size: 12px;
  font-weight: 800; letter-spacing: 1.5px;
}
#app-header h1 {
  position: relative; z-index: 1; margin: 0 0 8px; color: white;
  font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 900; line-height: 1.08; letter-spacing: -1.3px;
}
#app-header p { position: relative; z-index: 1; max-width: 760px; margin: 0; color: rgba(255,255,255,.87); font-size: 1.02rem; line-height: 1.6; }
.gradio-container button { border-radius: 13px !important; transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important; }
.gradio-container button:hover { transform: translateY(-1px); filter: saturate(1.08); }
#convert-button, #publish-button { min-height: 50px; font-weight: 800; }
#convert-button { color: white !important; border: none !important; background: linear-gradient(100deg,#6158ed,#8772ff 56%,#26a9c3) !important; box-shadow: 0 9px 20px rgba(99,91,255,.23); }
#convert-button:hover { box-shadow: 0 13px 24px rgba(99,91,255,.30); }
#publish-button { border: 1px solid rgba(99,91,255,.35) !important; background: linear-gradient(100deg,#f0efff,#e8fbff) !important; }
.gradio-container .block { border-radius: 15px !important; }
@keyframes studioFadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes studioFloat { from { transform: translate3d(0,0,0) scale(1); } to { transform: translate3d(-18px,18px,0) scale(1.06); } }
@media (max-width:700px) {
  .gradio-container { padding: 8px !important; }
  #app-header { padding: 22px 20px; border-radius: 18px; }
  #app-header p { font-size: .92rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
"""
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="AI Art Studio") as demo:
    gr.HTML("""
    <div id="app-header">
      <div class="hero-kicker">LOCAL AI • CREATIVE WORKSHOP</div>
      <h1>AI Art Studio</h1>
      <p>Turn photographs into expressive artwork with pretrained neural models, style-specific adapters, and composition guidance — all processed on your computer.</p>
    </div>
    """)

    gr.Markdown("### 1 · Upload a photograph and select a style")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Content photograph", type="pil", sources=["upload"])
            style_selector = gr.Dropdown(
                choices=STYLES, value="Cyberpunk", label="Choose an art style"
            )
            style_reference = gr.Image(
                label="Style reference artwork (only for Reference Style Transfer)",
                type="pil",
                sources=["upload"],
                visible=False,
            )
            style_strength = gr.Slider(
                minimum=0.30, maximum=0.95, value=0.78, step=0.05,
                label="Art style strength",
                info="For diffusion/reference styles: higher values create stronger stylization but may change details. Fixed neural styles use their own trained strength.",
            )
            structure_strength = gr.Slider(
                minimum=0.25, maximum=0.95, value=0.70, step=0.05,
                label="Composition / edge preservation",
                info="Used for diffusion styles. Higher values preserve large contours; lower values allow freer stylization.",
            )
            quality_steps = gr.Slider(
                minimum=20, maximum=40, value=36, step=4,
                label="Diffusion quality steps",
                info="More steps usually take longer; used by diffusion-based styles, not the fixed feed-forward models.",
            )
            seed_input = gr.Number(
                value=-1,
                precision=0,
                label="Seed (-1 = random, any other integer = repeatable result)",
            )
            convert_button = gr.Button("Transform Image", variant="primary", elem_id="convert-button")

        with gr.Column():
            gr.Markdown("### 2 · Compare, download, or approve a showcase")
            original_preview = gr.Image(label="Original", type="pil")
            result_preview = gr.Image(label="Transformed result", type="pil")

    download_file = gr.File(label="Download result", interactive=False)
    status = gr.Textbox(label="Status", value="Upload a photograph and choose a style.", interactive=False)

    gr.Markdown("## Publish a showcase image to GitHub")
    gr.Markdown(
        "Review the result, tick the approval box and type `PUBLISH`. Only the latest approved "
        "result is copied to `examples/`; input photos and `outputs/` are not staged by this button."
    )
    approve_checkbox = gr.Checkbox(
        label="I reviewed the result and approve publishing this image publicly.",
        value=False,
    )
    confirmation = gr.Textbox(label="Type PUBLISH to confirm", placeholder="PUBLISH")
    publish_button = gr.Button("Publish approved image to GitHub", variant="secondary", elem_id="publish-button")
    publish_status = gr.Markdown("No image has been published during this session.")

    def show_style_reference(style_name):
        return gr.update(visible=(style_name == "Reference Style Transfer"))

    style_selector.change(
        fn=show_style_reference,
        inputs=[style_selector],
        outputs=[style_reference],
    )

    convert_button.click(
        fn=transform_image,
        inputs=[
            input_image,
            style_selector,
            style_reference,
            style_strength,
            structure_strength,
            quality_steps,
            seed_input,
        ],
        outputs=[original_preview, result_preview, download_file, status],
    )
    publish_button.click(
        fn=publish_showcase,
        inputs=[approve_checkbox, confirmation],
        outputs=[publish_status],
    )

    gr.Markdown("""
    ---
    ### Models and honest limitations

    **Dedicated neural models:** AnimeGANv2; Starry Night fast-style transfer; Candy, Mosaic,
    Rain Princess and Udnie checkpoints where their published weight files download and load.

    **Diffusion-based styles:** Pop Art, Art Nouveau, Ukiyo-e, Impressionism, Cyberpunk, Surrealism,
    Art Deco, Watercolor, Pencil Sketch, Oil Painting and Pixel Art use Stable Diffusion image-to-image
    with Canny ControlNet. Where a compatible LoRA can load, the app adds it; if not, it visibly reports
    a warning and still attempts the style-specific prompt + ControlNet path instead of failing immediately.

    **Quality and accuracy:** there is no objectively valid “95% style-transfer accuracy” metric for this
    task. The app cannot guarantee identical faces, exact text, or preservation of every object. Higher
    style strength creates a more artistic result but increases the risk of changes; higher composition
    guidance preserves large contours. Always inspect the image before sharing it.

    **Local privacy:** uploaded photos and outputs are stored locally. Only an image you explicitly approve
    with the publishing controls is copied to the public `examples/` folder and pushed to GitHub.
    Model downloads happen from Hugging Face on first use and are cached locally; the app uses no paid API.
    """)


if __name__ == "__main__":
    # Local PC-only by default. For phone access on the same private Wi-Fi:
    # PowerShell: $env:AI_ART_STUDIO_HOST="0.0.0.0"; python app.py
    host = os.environ.get("AI_ART_STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("AI_ART_STUDIO_PORT", "7860"))
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("Note: first use of a diffusion style may download several GB of model weights.")
    demo.queue(max_size=2).launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        share=False,
        show_error=True,
    )
