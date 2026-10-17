# AI Art Studio

**Author:** Mohammed Naeem

A local-first image-stylization app built with Python, Gradio, PyTorch, Diffusers, Stable Diffusion, ControlNet, and selected pretrained style models.

Upload a photograph, select an artistic style, adjust the transformation settings, preview the result, and download it. The application runs inference on your computer; it does not require a paid image-generation API. An internet connection is needed for the initial package and model downloads.

> **Honest quality statement:** there is no universal “95% accuracy” score for artistic style transfer. This is image generation, not a classification task with a single correct answer. Results depend on the source image, model, adapter, settings, software versions, and available hardware. The app cannot guarantee identical faces, exact lettering, or preservation of every small object. Inspect outputs before using or sharing them.

---

## Contents

1. [Features](#1-features)
2. [What you need](#2-what-you-need)
3. [Install from GitHub on a fresh Windows PC](#3-install-from-github-on-a-fresh-windows-pc)
4. [Run the app](#4-run-the-app)
5. [Available styles and how they work](#5-available-styles-and-how-they-work)
6. [Recommended settings](#6-recommended-settings)
7. [Run the app from a phone](#7-run-the-app-from-a-phone)
8. [Update the app with Git](#8-update-the-app-with-git)
9. [Local files, privacy, and publishing examples](#9-local-files-privacy-and-publishing-examples)
10. [Gallery and sample outputs](#10-gallery-and-sample-outputs)
11. [Troubleshooting](#11-troubleshooting)
12. [Models, access, and licensing](#12-models-access-and-licensing)
13. [Known limitations](#13-known-limitations)

---

## 1. Features

- Gradio browser interface.
- Original image and transformed-result previews.
- Downloadable result images.
- Multiple style-specific paths: fixed-style neural models, Stable Diffusion image-to-image with Canny ControlNet, style LoRAs where compatible, and experimental VGG19 reference style transfer.
- Controls for art-style strength, composition/edge preservation, diffusion steps, and random or repeatable seed values.
- Local result storage in `outputs/`.
- Optional manual GitHub publishing for a single reviewed image at a time.
- Local-only mode by default; optional access over a trusted private Wi-Fi network.

The main application file is `app.py`. The Python package versions are constrained in `requirements.txt` to reduce known compatibility problems between Diffusers, Transformers, PEFT, and Hugging Face Hub.

## 2. What you need

- **Git** to download and update the source.
- **Python 3.12 recommended** for a predictable clean setup with the dependency set in this repository. Other Python versions may work, but are not guaranteed by this guide.
- **Internet access** for first-time package and model downloads.
- An NVIDIA GPU is recommended for diffusion-based styles. CPU fallback may work, but diffusion generation can be very slow.
- Several gigabytes of free disk space. More styles/models mean more model weights and cache files.
- Patience for the first run: downloading and loading a base model, ControlNet, and a style adapter can take time.

The app selects CUDA when PyTorch reports that CUDA is available; otherwise, it uses CPU. An Apple Silicon/MPS-specific acceleration path is not explicitly configured in the current app.

---

## 3. Install from GitHub on a fresh Windows PC

**You do not need your old project folder.** If you delete the local folder or replace your computer, you can recover the source from GitHub. The repository does not restore private photos, generated images, a virtual environment, or previously downloaded model weights; back up those separately if you want to keep them.

### Step A — Install Git and Python

1. Install Git for Windows: https://git-scm.com/download/win
2. Install Python 3.12: https://www.python.org/downloads/windows/
3. If the PC has an NVIDIA GPU, install/update its NVIDIA driver.
4. Open a new PowerShell terminal, Windows Terminal, or the VS Code terminal.

Check the installations:

```powershell
git --version
py -3.12 --version
```

If `py -3.12` is not found, install Python 3.12 and reopen the terminal. `py -0p` lists Python versions registered with the Windows launcher.

### Step B — Download the project from GitHub

These commands use your account's home directory; they do not hard-code a particular Windows username:

```powershell
cd $HOME
git clone https://github.com/mohammednaeemog-sys/AI-Art-Style-Transfer.git
cd AI-Art-Style-Transfer
```

The repository is now in a new `AI-Art-Style-Transfer` directory under your home folder. You can clone it elsewhere if you prefer; continue the remaining commands from inside the cloned repository.

If the repository already exists at that location, do not clone over it. Open the existing directory or choose a different parent directory.

### Step C — Create a fresh virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

A virtual environment keeps this application's Python packages separate from other projects. You do not have to activate it if you use the explicit `.venv\Scripts\python.exe` path in the commands below.

### Step D — Install a suitable PyTorch build

PyTorch needs to match the PC's operating system and compute hardware.

1. Open the official PyTorch selector: https://pytorch.org/get-started/locally/
2. Choose Windows, Pip, Python, and the correct compute platform.
3. Copy the command it provides for your PC. For an NVIDIA GPU, use a CUDA option supported by the driver and current PyTorch wheels. For a CPU-only PC, choose CPU.
4. Run that command using the new virtual environment's Python/pip. If the selector provides a command starting with `pip install`, use the same package and index arguments but invoke pip as:

```powershell
.\.venv\Scripts\python.exe -m pip install
```

For example, append `torch torchvision` and the exact `--index-url` or other arguments printed by the official selector. Do **not** copy an example CUDA index URL from an unrelated PC or blindly use a command from an old tutorial.

### Step E — Install the project's remaining requirements

From inside the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then check whether PyTorch detects CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

On an NVIDIA PC, `CUDA available` should normally be `True`. If it is `False`, revisit the official PyTorch selector and correct the installation in `.venv` before diagnosing application-level performance.

### Step F — Start AI Art Studio

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for Gradio to print a local URL, normally `http://127.0.0.1:7860`, and open the exact URL shown. Keep the terminal running while using the app. Press **Ctrl+C** in that terminal to stop it.

The first use of some styles may download several gigabytes of model files. This is expected; do not close the app while a download is in progress.

---

## 4. Install from GitHub on a fresh macOS or Linux PC

The Windows instructions above are the most direct path for the original development setup. On macOS or Linux, install Git and a compatible Python version using the operating system's official installer/package instructions, then run:

```bash
git clone https://github.com/mohammednaeemog-sys/AI-Art-Style-Transfer.git
cd AI-Art-Style-Transfer
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
```

Install PyTorch using the official selector for your OS/hardware: https://pytorch.org/get-started/locally/ . Then install the project requirements and launch:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app.py
```

Open the local URL printed by Gradio. NVIDIA GPU acceleration depends on compatible hardware, drivers, and a suitable PyTorch build. The app's current device selection uses CUDA when available and otherwise CPU; an Apple Silicon/MPS-specific path is not explicitly configured, so Apple GPU acceleration is not guaranteed.

For phone access from a Mac/Linux host on a trusted private Wi-Fi network, start the app with:

```bash
AI_ART_STUDIO_HOST=0.0.0.0 AI_ART_STUDIO_PORT=7860 ./.venv/bin/python app.py
```

Find the host computer's local IP address using `ip addr` on Linux or `ifconfig`/System Settings on macOS, then open `http://<HOST-LOCAL-IP>:7860` on the phone while both devices are on the same trusted Wi-Fi. Do not expose the app to an untrusted or public network.

---

## 4. Run the app

After the first installation, you do not need to reinstall packages every time.

In PowerShell:

```powershell
cd $HOME\AI-Art-Style-Transfer
.\.venv\Scripts\python.exe app.py
```

If you cloned the repository to a different location, `cd` to that location instead.

In VS Code:

1. Choose **File → Open Folder**.
2. Open the cloned `AI-Art-Style-Transfer` folder.
3. Choose **Terminal → New Terminal**.
4. Run `.\.venv\Scripts\python.exe app.py`.
5. Open the local URL printed in the terminal.

If the terminal reports that port 7860 is already in use, choose another port:

```powershell
$env:AI_ART_STUDIO_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

Open the new URL printed in the terminal.

---

## 5. Available styles and how they work

The selector includes 18 options. The model path may vary if a specific checkpoint cannot be downloaded or loaded.

| Style | Intended model/method | What to expect |
|---|---|---|
| **Anime** | AnimeGANv2; diffusion fallback if unavailable | Anime-like appearance; face and object details can change. |
| **Starry Night (Neural)** | Dedicated fixed-style feed-forward model; prompt/ControlNet fallback if unavailable | Van Gogh-inspired colors and painted texture. |
| **Candy (Neural)** | Fixed-style fast-neural-style checkpoint; fallback if unavailable | Vivid painterly textures associated with the checkpoint. |
| **Mosaic (Neural)** | Fixed-style fast-neural-style checkpoint; fallback if unavailable | Mosaic-like texture from the checkpoint. |
| **Rain Princess (Neural)** | Fixed-style fast-neural-style checkpoint; fallback if unavailable | Colorful painterly effect from the checkpoint. |
| **Udnie (Neural)** | Fixed-style fast-neural-style checkpoint; fallback if unavailable | The checkpoint's specific abstract/painterly style. |
| **Pop Art** | Stable Diffusion 1.5 + Canny ControlNet + compatible Pop Art LoRA when it loads | Comic-book shapes, stronger colors and printmaking-inspired details. |
| **Art Nouveau** | SDXL + SDXL Canny ControlNet + Mucha LoRA when compatible | Decorative botanical and poster-like illustration. SDXL is more memory-demanding. |
| **Ukiyo-e** | Stable Diffusion 1.5 + Canny ControlNet + Hiroshige LoRA when compatible | Japanese woodblock-print-inspired colors and linework. |
| **Impressionism** | Stable Diffusion 1.5 + Canny ControlNet + Monet LoRA when compatible | Painterly colors and brushwork. |
| **Cyberpunk** | Cyberpunk-focused checkpoint + Canny ControlNet; baseline prompt fallback if needed | Neon/futuristic styling; exact model compatibility matters. |
| **Surrealism** | Stable Diffusion 1.5 + Canny ControlNet + style-specific prompt | Dreamlike atmosphere; not guaranteed to invent specific objects. |
| **Art Deco** | Stable Diffusion 1.5 + Canny ControlNet + style-specific prompt | Geometric, vintage-poster-inspired styling. |
| **Watercolor** | Stable Diffusion 1.5 + Canny ControlNet + style-specific prompt | Watercolor-inspired washes and edges. |
| **Pencil Sketch** | Stable Diffusion 1.5 + Canny ControlNet + style-specific prompt | Graphite/monochrome drawing style. |
| **Oil Painting** | Stable Diffusion 1.5 + Canny ControlNet + compatible oil-painting LoRA when it loads | Painted surfaces and brushstroke-like texture. |
| **Pixel Art** | Stable Diffusion 1.5 + Canny ControlNet + compatible PixelArtRedmond LoRA when it loads | Pixel-art-inspired illustration. |
| **Reference Style Transfer** | Experimental VGG19 neural optimization using your uploaded style artwork | Transfers texture/color statistics; can look noisy and does not semantically recreate the reference image. |

### Important: fallback is not the same as success

For diffusion-based styles, the app tries to load the configured base checkpoint, ControlNet, and compatible adapter. If a preferred model cannot load, it attempts a fallback path and reports a warning in the status message or terminal. A fallback can generate an image, but it may be less faithful to the intended style. If the status reports that the dedicated model failed, do not assume the dedicated model produced the result.

---

## 6. Recommended settings

These are **starting points**, not proven universal optima.

- **Art style strength:** start around `0.70–0.82`. Increase it for a stronger artistic transformation; lower it if the subject changes too much.
- **Composition / edge preservation:** start around `0.65–0.80`. Higher values usually preserve broad outlines more strongly; lower values allow more freedom.
- **Diffusion quality steps:** start around `32–36`. More steps take longer and do not guarantee a better result.
- **Seed:** `-1` selects a random seed. Enter a non-negative integer to try to repeat a comparable diffusion run with the same model and settings.

For portraits, start with lower style strength (about `0.60–0.72`) and higher structure guidance (about `0.75–0.85`). For landscapes or graphic art, a stronger transformation may work better. Different models respond differently, so compare results rather than assuming one setting will always be best.

---

## 7. Run the app from a phone

The application runs on your PC; the phone opens the interface in its browser. Both devices must be connected to the **same trusted private Wi-Fi network**.

1. Start PowerShell in the project root on the PC.
2. Set the network host and port, then start the application:

```powershell
$env:AI_ART_STUDIO_HOST="0.0.0.0"
$env:AI_ART_STUDIO_PORT="7860"
.\.venv\Scripts\python.exe app.py
```

3. On the PC, open another terminal and run:

```powershell
ipconfig
```

4. Find the IPv4 address of the active Wi-Fi adapter, for example `192.168.1.23`.
5. If Windows Firewall asks, allow Python on **Private networks only**.
6. Connect the phone to the same Wi-Fi. In its browser, enter the PC's actual IPv4 address and port, for example:

```text
http://192.168.1.23:7860
```

Replace the example IP with the one shown by `ipconfig`. Do not type `127.0.0.1` on the phone; that would point to the phone itself.

Keep the app running on the PC. The PC does the model inference and must remain powered on.

**Security:** `0.0.0.0` makes the app reachable from devices that can access the PC on that network. Use only a trusted private Wi-Fi network. Do not configure router port forwarding or expose this unauthenticated app to the public internet. Stop the app with Ctrl+C when finished. Environment-variable settings in the current PowerShell session do not persist after you close that shell.

---

## 8. Update the app with Git

When you have an existing clone, use `git pull` to download committed changes. You do not need to clone the project again for every update.

First, go to the repository folder and check for local edits:

```powershell
cd $HOME\AI-Art-Style-Transfer
git status
```

If you have uncommitted work, back it up or commit it before pulling. Then run:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If the update changes a major dependency, reread the setup notes and verify CUDA afterwards.

If `git pull --ff-only` fails, stop and inspect `git status`, `git remote -v`, and `git log --oneline --decorate -5`. Do not force-push or discard local changes just to clear an error.

### Publish a code/documentation update

Before committing, inspect the list of files:

```powershell
git status --short
git add app.py requirements.txt README.md .gitignore
git diff --cached --name-only
git diff --cached --check
```

Only stage files you intend to publish. Do not commit input photographs, datasets, downloaded weights, private outputs, credentials, or `.env` files. After reviewing the list, commit and push:

```powershell
git commit -m "Update AI Art Studio documentation"
git push origin main
```

---

## 9. Local files, privacy, and publishing examples

### Files stored locally

- Input photographs are processed by the app on your computer.
- Generated images are saved to the local `outputs/` directory.
- Downloaded model files are cached locally by their libraries and/or the project's ignored model directory.
- The `.venv` folder contains your local Python environment.

The `.gitignore` is designed to exclude the virtual environment, common input image filenames, `outputs/`, data folders and model files from **future untracked commits**. Git ignore rules do not remove files that were committed previously. If any private image or model file was committed in the past, inspect GitHub history and remove it from the repository/history as needed.

### Publish one approved example

The app includes an explicit publishing workflow:

1. Generate a result.
2. Inspect it and confirm you have the right to share it.
3. Tick the approval checkbox.
4. Type `PUBLISH`.
5. Click **Publish approved image to GitHub**.

The button is designed to copy only the latest result to `examples/`, stage only that image, commit it and push to the configured `origin` remote. If unrelated files are already staged, the publishing function is intended to stop rather than commit them. Publishing requires Git to be installed and GitHub push authentication to be configured.

The repository is public. Anyone may view or copy a showcase image after publication. Never publish private photos, identifiable portraits without permission, or an image whose license prohibits sharing.

---

## 10. Gallery and sample outputs

No generated image should be presented as a model output until the app has actually generated it. This repository may have no examples in a fresh clone because sample results are optional and only reviewed outputs should be published.

After you approve an output, the publishing button places it in `examples/`. You can then add a gallery entry here, using the **actual filename** of the image committed to GitHub. For example:

```markdown
## Showcase

### Cyberpunk
![Approved Cyberpunk output](examples/your-actual-cyberpunk-filename.png)

### Impressionism
![Approved Impressionism output](examples/your-actual-impressionism-filename.png)
```

Replace those example filenames with files that really exist in the repository. Do not leave a sample entry that points to a nonexistent file and do not label a mock/generated illustration as if it came from the application.

---

## 11. Troubleshooting

### Python launcher cannot find Python 3.12

Install Python 3.12 from https://www.python.org/downloads/windows/, reopen PowerShell, then try `py -3.12 --version`. `py -0p` lists available Python installations.

### PowerShell blocks environment activation

Activation is optional. Run the app through the virtual environment's Python executable:

```powershell
.\.venv\Scripts\python.exe app.py
```

If you prefer activation, run the following in the current PowerShell process, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Missing Python module or dependency import error

Confirm that the terminal is in the repository root, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Restart the app. Avoid upgrading Diffusers, Transformers, PEFT, and Hugging Face Hub independently; incompatible versions can break adapter loading.

### CUDA is unavailable

Run the PyTorch/CUDA verification command in Step E. If a GPU is present but CUDA reports `False`, reinstall the correct PyTorch build using the official selector: https://pytorch.org/get-started/locally/

### CUDA out of memory or generation is too slow

Close other GPU-heavy applications, reduce image size or diffusion steps, and avoid the SDXL-based Art Nouveau route if memory is limited. CPU mode is slower. A GPU with 8 GB VRAM may need CPU offloading and can still be slow or hit memory limits depending on the model and software configuration.

### Model download fails or returns 401/403

Check the internet connection and the exact model card. Some model providers require sign-in, agreement to model terms, or permission. Resolve those requirements on the provider's site. Never paste access tokens into source code, public screenshots, or chat messages.

### Style adapter fails to load

Read the status message and terminal. The app may attempt a prompt + ControlNet fallback; that does not mean the dedicated adapter worked. Record the selected style and complete error text before troubleshooting.

### Local page does not open

Keep the app terminal running and open the exact URL it prints. If port 7860 is occupied, set `AI_ART_STUDIO_PORT` to a free port such as `7861` before launching. For phone access, use the PC's local IPv4 address and the same port.

### Fresh clone cannot find a previous image

That is expected. `git clone` downloads committed files only. It will not recover ignored local photos, `outputs/`, `.venv`, or cached model weights from your old PC. Keep separate backups of any local file you want to preserve.

---

## 12. Models, access, and licensing

Model weights are downloaded from third-party publishers and are not bundled with the source repository. A model being publicly downloadable does not mean all uses, redistribution, or commercial use are permitted. Review each model's current card and license before distributing this application, weights, or generated results.

- [Stable Diffusion 1.5 base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Canny ControlNet for SD 1.5](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)
- [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Canny ControlNet for SDXL](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)
- [Pop Art LoRA](https://huggingface.co/DesiAEye/DesiAEye-Pop-Art-LORA)
- [Alphonse Mucha / Art Nouveau LoRA](https://huggingface.co/KappaNeuro/alphonse-mucha-style)
- [Hiroshige / Ukiyo-e LoRA](https://huggingface.co/espressosession/latent-studio-hiroshige-lora)
- [Monet / Impressionism LoRA](https://huggingface.co/J-YOON/lora-monet-sd1.5)
- [Cyberpunk diffusion checkpoint](https://huggingface.co/AdamOswald1/Cyberpunk-Anime-Diffusion_with_support_for_Gen-Imp_characters)
- [Oil-painting LoRA](https://huggingface.co/Jehugging/oilpaint_lora)
- [PixelArtRedmond LoRA](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5)
- [Starry Night fast-style model](https://huggingface.co/Rohanify/Brawnz-StyleTransferSN)
- [PyTorch fast-neural-style examples](https://github.com/pytorch/examples/tree/main/fast_neural_style)

Model files, filenames, access policies, licenses and compatibility may change. The application may need updates if a third-party repository changes or a particular checkpoint no longer loads.

The repository source itself should also have an explicit `LICENSE` file if you want other people to know what permissions apply to reusing your code. Do not assume a permissive project-code license merely because the repository is public.

---

## 13. Known limitations

- The intended style adapter can fail to load due to model availability, access rules, dependency compatibility, or base-model incompatibility. The app attempts fallbacks for some paths; fallback quality may be lower.
- The reference VGG19 path is slow and may transfer texture without producing a convincing semantic recreation of the reference artwork.
- Canny ControlNet helps guide broad structure but does not guarantee exact object/face/text preservation.
- A stronger style setting may improve the artistic effect while reducing source-image fidelity.
- CPU mode can be extremely slow for diffusion.
- The complete set of model combinations has not been verified on every Windows/macOS/Linux system, Python version, and GPU. Verify each selected style in the environment where you intend to use it before calling the setup complete.


