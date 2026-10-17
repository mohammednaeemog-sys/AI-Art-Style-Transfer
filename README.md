# AI Art Studio

A **free, locally running** image-stylization application built with Python, Gradio, PyTorch, Diffusers, Stable Diffusion, ControlNet, and style adapters (LoRAs). It has an animated web interface, image preview/download, adjustable composition/style controls, and an opt-in GitHub showcase publisher.

> **Straight truth about quality:** there is no universal “95%+ accuracy” metric for artistic style transfer. This is not a classifier with a correct/incorrect answer; the quality is subjective and varies by input image and model. The app is hardened to avoid known LoRA-loading failures and to show a visible fallback status, but it cannot promise a perfect image or identical faces, text, and small objects on every run. You must inspect the result.

## Contents

- [Features and model routes](#features-and-model-routes)
- [Files and outputs](#files-and-outputs)
- [Install/update on the current Windows PC](#installupdate-on-the-current-windows-pc)
- [Publish code to GitHub](#publish-the-code-to-github)
- [Install on another PC](#install-on-another-pc)
- [Open the app from a phone](#open-the-app-from-a-phone)
- [Quality settings guide](#quality-settings-guide)
- [Troubleshooting](#troubleshooting)
- [Models, sources, and licenses](#models-sources-and-licenses)
- [Known limitations](#known-limitations)

---

## Features and model routes

The app routes each choice to the most suitable available path in this local setup.

| Style | Model/method | What to expect |
|---|---|---|
| **Anime** | AnimeGANv2 feed-forward neural model | Fast photo-to-anime appearance; can change small details. |
| **Starry Night (Neural)** | Dedicated feed-forward Starry Night model | Fast Van Gogh-inspired colors/texture; uses diffusion fallback if the checkpoint cannot load. |
| **Candy (Neural)** | Published fast-neural-style checkpoint | Fixed painterly style; fallback if download or state-dict loading fails. |
| **Mosaic (Neural)** | Published fast-neural-style checkpoint | Mosaic-like texture; fallback if the checkpoint cannot load. |
| **Rain Princess (Neural)** | Published fast-neural-style checkpoint | Soft colorful painterly style; fallback if the checkpoint cannot load. |
| **Udnie (Neural)** | Published fast-neural-style checkpoint | Abstract modernist color shapes; fallback if the checkpoint cannot load. |
| **Pop Art** | Stable Diffusion 1.5 + Canny ControlNet + Pop Art LoRA when it loads | Comic-book/printmaking-inspired stylization, bold colors and contours. |
| **Art Nouveau** | SDXL + SDXL Canny ControlNet + Alphonse Mucha LoRA when it loads | Ornamental, botanical, vintage decorative illustration. SDXL can be slower and heavier. |
| **Ukiyo-e** | SD 1.5 + Canny ControlNet + Hiroshige LoRA when it loads | Japanese woodblock-print-inspired color and linework. |
| **Impressionism** | SD 1.5 + Canny ControlNet + Monet LoRA when it loads | Painterly brushwork and impressionist colors. |
| **Cyberpunk** | Cyberpunk-focused Stable Diffusion checkpoint + Canny ControlNet; SD 1.5 fallback | Neon, cinematic, futuristic stylization. The selected checkpoint may be more anime-oriented than photorealistic. |
| **Surrealism** | SD 1.5 + Canny ControlNet + style prompt | Dreamlike fine-art look. A prompt cannot reliably invent or place exact objects. |
| **Art Deco** | SD 1.5 + Canny ControlNet + style prompt | Geometric poster look. A model-card adapter that was not a standard Diffusers LoRA is deliberately not loaded. |
| **Watercolor** | SD 1.5 + Canny ControlNet + style prompt | Watercolor-inspired washes and soft edges. |
| **Pencil Sketch** | SD 1.5 + Canny ControlNet + style prompt | Graphite/monochrome sketch effect. |
| **Oil Painting** | SD 1.5 + Canny ControlNet + oil-paint LoRA when it loads | Painterly surfaces and brush strokes. |
| **Pixel Art** | SD 1.5 + Canny ControlNet + PixelArtRedmond LoRA when it loads | Pixel-art illustration. The model's license is nonstandard; check it before redistribution/commercial use. |
| **Reference Style Transfer** | Experimental VGG19 optimization with a style artwork uploaded by the user | Transfers texture/color statistics; it can be noisy and does not semantically recreate the reference composition. |

For diffusion styles, the app uses **image-to-image + Canny ControlNet**. The original image supplies the content image and an edge map guides broad contours. A compatible LoRA adds the learned style. Diffusers documents this image-to-image/ControlNet pattern in its [ControlNet image-to-image pipeline example](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py).

**Important fallback behavior:** an adapter that fails to load no longer automatically stops the transformation with a blank image. The status field tells you the adapter failed and that the app used the style-specific prompt + ControlNet route instead. The full error is printed in the terminal. If even the base model/ControlNet cannot load, the app returns a real error instead of pretending success.

### Why results differ

- **Higher Art style strength** lets the diffusion model change more of the photo, so the style is stronger but some details may change.
- **Higher Composition / edge preservation** helps retain large contours; it does not guarantee the same face, text, or tiny objects.
- **More diffusion steps** can improve refinement but take longer; steps are not a guarantee of quality.
- **Reference Style Transfer** does not use a diffusion model. It is an experimental optimization baseline and can produce the noisy texture overlay you saw previously. Prefer a dedicated style preset when one exists.

---

## Files and outputs

```text
AI-Art-Style-Transfer/
├── app.py                  # Main web application
├── requirements.txt        # Compatible Python dependency versions
├── README.md               # This guide
├── .gitignore              # Excludes local photos, outputs, model files and secrets
├── outputs/                # Generated images; local only, ignored by Git
├── models/                 # Download cache used by some local neural models; ignored
└── examples/               # Public showcase images you manually approve
```

The application also uses Hugging Face and PyTorch caches in your Windows user profile, typically under:

- `%USERPROFILE%\.cache\huggingface\hub`
- `%USERPROFILE%\.cache\torch\hub`

Generated images are stored locally in `outputs/` with a timestamp and style name, e.g. `20261017_152400_123456_cyberpunk.png`. The results appear in the browser and can be downloaded immediately.

**No fake sample outputs are bundled in this repository.** The `examples/` folder is reserved for real results that you review and explicitly approve. Add an actual approved example through the app's publishing controls; do not label an illustrative/mock image as a model output.

---

## Install/update on the current Windows PC

Your current folder is:

```text
C:\Users\User\AI-Art-Style-Transfer
```

### A. Replace the code safely

1. In VS Code, click the terminal running the app and press **Ctrl+C**.
2. Back up your current `app.py` somewhere **outside** the repository (for example, your Desktop).
3. Extract the downloaded project ZIP into a temporary folder.
4. Copy these files from the extracted folder into your project root and replace them:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
5. Keep your existing `.venv`, input photos, datasets, and output files. Do not put private photos in `examples/`.
6. In VS Code, confirm that the terminal path is the project root:

```powershell
cd C:\Users\User\AI-Art-Style-Transfer
```

### B. Install a compatible dependency set

If the terminal prompt already starts with `(.venv)`, the virtual environment is active. Do not type square brackets around the command.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements intentionally pin a compatible Diffusers/Transformers/PEFT set, because mixing a newer Transformers release with an older model-loading stack can make LoRA loading unreliable.

Check PyTorch and CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

For the 8 GB NVIDIA GPU, `CUDA available` should print `True`. If it says `False`, install the matching PyTorch CUDA build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). The known matching command for **PyTorch 2.11 + CUDA 12.8 on Windows** is:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Do this PyTorch install first, then run `python -m pip install -r requirements.txt` again. For a different GPU, operating system or CUDA choice, use the command generated by the official PyTorch selector instead of blindly copying that CUDA command.

If PowerShell blocks activation and you need to activate the environment manually:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### C. Start the app

```powershell
python app.py
```

Open the exact local URL printed in the terminal, normally:

```text
http://127.0.0.1:7860
```

To stop the application, click the terminal and press **Ctrl+C**.

### D. First-run downloads

The first use of a diffusion style downloads the relevant base model, ControlNet and (when configured) style adapter from Hugging Face. Depending on which styles you use, allow **many gigabytes of disk space**; SDXL uses additional space. Keep an internet connection until each chosen model is downloaded. Later runs use local caches.

A diffusion transformation at 512px and 32–36 steps can take minutes on an 8 GB laptop GPU with CPU offload. CPU-only operation is supported by the code but diffusion will be substantially slower. Close other GPU-heavy apps if you see an out-of-memory message.

---

## Publish the code to GitHub

The files downloaded in this chat have **not** been pushed into your GitHub repository automatically. To put this version in your current clone, first finish the replacement steps above, then run:

```powershell
git status --short
```

Review the list. It must not contain personal photos, model weights, `.env` files, or generated `outputs/`. Then stage only the intended code/docs:

```powershell
git add app.py requirements.txt README.md .gitignore
git diff --cached --name-only
```

Confirm the staged list contains only those approved text files (and any other code file you intentionally edited). Do **not** run `git add .` for this project.

Commit and push:

```powershell
git commit -m "Improve style models, fallback handling, and UI"
git push origin main
```

If GitHub says the push was rejected because your branch is behind, do not force push. Inspect `git status`, back up local work, then run `git pull --ff-only origin main` and resolve any legitimate conflict before retrying.

### Keep showcase images separate

Use the app's **Publish an approved image to GitHub** controls:
1. Generate an output and inspect it.
2. Tick the approval checkbox.
3. Type `PUBLISH`.
4. Click **Publish approved image to GitHub**.

This action copies only the latest image you approve to `examples/`, stages only that file, commits it, and attempts to push it. Inputs and other images under `outputs/` stay local. The repository is public, so only publish images you have permission to share.

---

## Install on another PC

To clone and run the project on a second Windows PC, the latest code must first have been committed and pushed to GitHub from your current PC.

### A. Install prerequisites

1. Install [Git for Windows](https://git-scm.com/download/win).
2. Install Python 3.12 or 3.13 from [python.org](https://www.python.org/downloads/windows/) (enable the launcher/`py` command during install).
3. Install/update the NVIDIA driver if the second PC has an NVIDIA GPU.
4. Open PowerShell.

### B. Clone the project

```powershell
cd $HOME\Documents
git clone https://github.com/mohammednaeemog-sys/AI-Art-Style-Transfer.git
cd AI-Art-Style-Transfer
```

If Python 3.12 is installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If only Python 3.13 is installed, replace `py -3.12` with `py -3.13`.

If PowerShell refuses activation, use the process-scoped policy command from the prior section, then retry activation.

### C. Install PyTorch and app dependencies

On an NVIDIA PC, choose the matching OS/Pip/Python/CUDA command from the [official PyTorch selector](https://pytorch.org/get-started/locally/). On a CPU-only PC, choose CPU. Then run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python app.py
```

Open the localhost URL. The new PC creates its own model caches and downloads models on first use; weights are not stored in GitHub. The first run needs internet access and enough disk space. If a model repository requires acceptance of its terms, open that model's card, review/accept the terms while signed in, and retry. Never paste access tokens into code or chat.

### D. Update an existing clone later

From inside the repository:

```powershell
git status
git pull --ff-only origin main
python -m pip install -r requirements.txt
python app.py
```

If `git status` shows modified files you want to keep, commit/backup them before pulling. `git pull` updates code; it does not copy model weights or local photos from the original computer.

---

## Open the app from a phone

The AI runs on the **PC**; the phone is only the browser control/display. Both devices must be on the same trusted Wi-Fi/LAN. This is not cloud hosting.

1. Stop the app with **Ctrl+C** if it is currently running.
2. On the PC, open PowerShell in the project root and activate `.venv`.
3. Start the app bound to the local network interface:

```powershell
$env:AI_ART_STUDIO_HOST="0.0.0.0"
$env:AI_ART_STUDIO_PORT="7860"
python app.py
```

4. On the PC, run `ipconfig` and find the active Wi-Fi adapter's **IPv4 Address**, for example `192.168.1.23`.
5. If Windows Firewall asks, allow Python for **Private networks only**. Do not open the app to public networks.
6. On the phone, connect to the same Wi-Fi and visit:

```text
http://192.168.1.23:7860
```

Replace the example IP with the PC's actual IPv4 address. Keep the terminal/app running on the PC while using the phone. Use `Ctrl+C` to stop it. To return to PC-only access, close the terminal or start it without `AI_ART_STUDIO_HOST`, which defaults to `127.0.0.1`.

**Security:** the local-network mode makes the interface available to devices that can reach the PC on that network. Do not router-port-forward the port, and do not use this mode on public Wi-Fi. The app is configured with `share=False`; it does not intentionally create a public Gradio tunnel.

---

## Quality settings guide

Good starting settings for **diffusion styles** on the 8 GB GPU:

- **Art style strength:** `0.70–0.82`
- **Composition / edge preservation:** `0.65–0.80`
- **Diffusion quality steps:** `32–36`
- **Seed:** `-1` for random output; use any non-negative integer to repeat a run (it may still vary across software/hardware versions)

For portraits or photos containing people, start with a slightly lower style strength (around `0.65–0.72`) and a higher structure value (around `0.75–0.85`). For a stronger painterly or graphic result, raise style strength gradually. Do not expect increasing the step count alone to fix the wrong style model.

If a style's adapter fails, read the status box. A `WARNING` means the app used the prompt+ControlNet fallback, not the failed LoRA. That makes the result available, but it may be less style-specific.

---

## Troubleshooting

### A style adapter reports a warning but still generates

That means the chosen LoRA could not be loaded or parsed. The app records the underlying error in the VS Code terminal and continues with the style prompt + ControlNet where possible. Check internet/model access and the pinned dependency set. Restart the app after correcting connectivity to retry an adapter previously marked failed during that run.

### `ImportError`, `No module named diffusers`, `peft`, or `accelerate`

Activate `.venv` and run:

```powershell
python -m pip install -r requirements.txt
```

Restart the app. Keep the compatible versions in `requirements.txt`; do not randomly upgrade one of Diffusers, Transformers, PEFT or Hugging Face Hub in isolation.

### `CUDA out of memory`

Close other programs using the GPU and retry. The code attempts a lower-resolution/step fallback on supported OOM paths. If memory errors continue, use a lower-resolution image and lower steps. SDXL Art Nouveau is heavier than SD 1.5 styles.

### HTTP 401/403 or model download errors

Check the terminal output and internet connection. Some model repositories may have special terms/access requirements; open the exact model card and accept its terms if required. Avoid posting authentication tokens in issues, screenshots or code.

### The local page does not open

Make sure the terminal still shows the running URL and hasn't returned to the PowerShell prompt. If port 7860 is occupied, run:

```powershell
$env:AI_ART_STUDIO_PORT="7861"
python app.py
```

Open the URL shown in the terminal. On a phone, confirm that both devices are on the same trusted Wi-Fi and use the PC's IPv4 address, not `127.0.0.1`.

### `git pull` fails

First run:

```powershell
git status
git remote -v
git branch --show-current
```

Keep or back up local changes before pulling. Do not force-push to resolve a rejected push without inspecting the history.

---

## Models, sources, and licenses

The app is free to run locally, but model licenses are **not all identical**. Review each model card and license before publishing outputs, redistributing model weights, or using outputs commercially.

- [Stable Diffusion 1.5 base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) — CreativeML OpenRAIL-M.
- [ControlNet Canny for SD 1.5](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) — OpenRAIL.
- [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- [SDXL Canny ControlNet](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0).
- [Pop Art LoRA](https://huggingface.co/DesiAEye/DesiAEye-Pop-Art-LORA) — check model card and CreativeML OpenRAIL-M terms.
- [Alphonse Mucha / Art Nouveau LoRA](https://huggingface.co/KappaNeuro/alphonse-mucha-style) — SDXL adapter; model card lists a nonstandard `other` license.
- [Hiroshige / Ukiyo-e LoRA](https://huggingface.co/espressosession/latent-studio-hiroshige-lora) — SD 1.5, CreativeML OpenRAIL-M.
- [Monet / Impressionism LoRA](https://huggingface.co/J-YOON/lora-monet-sd1.5) — SD 1.5, MIT.
- [Cyberpunk Anime Diffusion checkpoint](https://huggingface.co/AdamOswald1/Cyberpunk-Anime-Diffusion_with_support_for_Gen-Imp_characters) — model card describes an SD 1.5 fine-tune and OpenRAIL-M terms.
- [Oil-painting LoRA](https://huggingface.co/Jehugging/oilpaint_lora) — inspect current model-card terms.
- [PixelArtRedmond LoRA](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5) — bespoke LoRA license; review terms carefully.
- [Starry Night fast-style model](https://huggingface.co/Rohanify/Brawnz-StyleTransferSN) — MIT.
- [PyTorch fast-neural-style examples](https://github.com/pytorch/examples/tree/main/fast_neural_style).

Model repositories and their file names can change after this README is written. The app uses known filenames where checked and has a prompt+ControlNet fallback for adapter load errors, but no code can guarantee that external model hosting, package ecosystems, all hardware, and all images will behave perfectly forever.
