# AI Art Studio

**Author:** Mohammed Naeem

A free, local-first image-stylization app built with Python, Gradio, PyTorch, Diffusers, Stable Diffusion, ControlNet, and selected style adapters (LoRAs).

The web interface runs on your own computer. You upload a photo, choose a style, inspect the result, and download it. You can optionally publish a single image you have manually approved to the public GitHub repository.

> **Important quality note:** artistic style transfer does not have a universal “95% accuracy” score. This is image generation, not a classification task with a universally correct answer. Results vary by photo, selected model, model version, and settings. No method here guarantees identical faces, exact text, or preservation of every small object. Review each result before sharing it.

## Contents

- [1. What is included](#1-what-is-included)
- [2. The most important thing to know before reinstalling](#2-the-most-important-thing-to-know-before-reinstalling)
- [3. Install from GitHub on a completely fresh Windows PC](#3-install-from-github-on-a-completely-fresh-windows-pc)
- [4. Start the app](#4-start-the-app)
- [5. How model downloads work](#5-how-model-downloads-work)
- [6. Update the app later with git pull](#6-update-the-app-later-with-git-pull)
- [7. Use the app from a phone](#7-use-the-app-from-a-phone)
- [8. Run on macOS or Linux](#8-run-on-macos-or-linux)
- [9. Available styles and honest limitations](#9-available-styles-and-honest-limitations)
- [10. Local files, privacy, and GitHub publishing](#10-local-files-privacy-and-github-publishing)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Model sources and licenses](#12-model-sources-and-licenses)

---

## 1. What is included

The main files in the repository are:

```text
AI-Art-Style-Transfer/
├── app.py                 # Main application and browser interface
├── requirements.txt       # Python package requirements
├── README.md              # This guide
├── .gitignore             # Prevents local/private files from being committed
├── examples/              # Only manually approved public showcase images
├── outputs/               # Generated images; created locally and ignored by Git
└── models/                # Some local model files/cache; ignored by Git
```

Some of those folders are created automatically by the app when it starts. Empty folders or local cache folders might not appear in a fresh Git clone.

### What you need

- Git, to download and update the source repository.
- Python 3.12 recommended for a predictable setup with the pinned Diffusers stack. Python 3.13 may work with compatible PyTorch wheels, but on a fresh PC start with Python 3.12 if you can.
- Internet access for initial package and model downloads.
- An NVIDIA GPU is recommended for diffusion-based styles. CPU mode may work, but generation can be very slow.
- Several GB of free disk space; using multiple base models/adapters can require considerably more.

---

## 2. The most important thing to know before reinstalling

**You do not need the old local folder to reinstall the app.** The source code is stored in GitHub. On a new or wiped PC, clone the repository into any folder you choose, create a new Python virtual environment, install dependencies, and start the app. The commands below use `$HOME` (your account's home folder) instead of hard-coding a user-specific Windows folder.

However, GitHub only contains files that were committed and pushed. It does **not** contain your ignored private photos, generated `outputs/`, model caches, `.venv`, or downloaded model weights. If you delete your computer's files, local photos and generated results that you did not back up may be lost. Back them up separately if you need to keep them.

---

## 3. Install from GitHub on a completely fresh Windows PC

Follow this section if you have deleted the project folder, are setting up a different computer, or want to reinstall cleanly.

### Step A — Install prerequisites

1. Install **Git for Windows**: https://git-scm.com/download/win
2. Install **Python 3.12** from: https://www.python.org/downloads/windows/
   - During installation, enable the Python launcher (`py`) if the installer offers that option.
3. If the PC has an NVIDIA GPU, update its NVIDIA driver.
4. Open a new **PowerShell** terminal (Windows Terminal or the VS Code terminal is fine).

Check Git and Python:

```powershell
git --version
py -3.12 --version
```

If `py -3.12` is not found, install Python 3.12. To list Python versions installed on the PC, run `py -0p`.

### Step B — Clone the repository

These commands use your home directory, so they work without knowing your Windows username or hard-coding a machine-specific folder.

```powershell
cd $HOME
git clone https://github.com/mohammednaeemog-sys/AI-Art-Style-Transfer.git
cd AI-Art-Style-Transfer
```

The source is now downloaded into a new `AI-Art-Style-Transfer` folder inside your home directory. You can choose another destination; just `cd` into the cloned repository before continuing.

If Git asks you to authenticate, follow the GitHub sign-in prompts. Never put a GitHub token in `app.py`, `.env` files that are committed, or chat messages.

### Step C — Create a new virtual environment

This creates a clean Python environment just for this app:

```powershell
py -3.12 -m venv .venv
```

You do **not** have to activate the environment if you always use the `.venv` Python executable directly. That avoids common PowerShell activation-policy issues.

Upgrade pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Step D — Install PyTorch for this PC

Choose the PyTorch build for the hardware in the new computer:

- Open the official installer selector: https://pytorch.org/get-started/locally/
- Choose **Windows**, **Pip**, **Python**, and the correct compute platform.
- For an NVIDIA GPU, choose a CUDA option supported by the installed driver and available PyTorch wheels.
- For a CPU-only PC, choose the CPU build.

The selector gives a command similar to `pip3 install torch torchvision ...`. Run the equivalent command with this new environment's Python so it installs into `.venv`. For example, if the selector specifies an `--index-url`, keep it and use this command shape:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url <PASTE-THE-INDEX-URL-FROM-THE-PYTORCH-SELECTOR>
```

Replace the placeholder with the exact index URL from the official selector. Do not paste the angle brackets literally, and do not blindly use a CUDA command intended for a different PC.

### Step E — Install the app's remaining dependencies

From inside the cloned repository:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file keeps the Diffusers/Transformers/PEFT versions in a compatible set. Avoid separately upgrading one of those packages without checking compatibility.

Verify the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

On an NVIDIA PC, `CUDA available` should normally be `True`. If it is `False`, revisit the PyTorch selector and reinstall the correct build into `.venv`.

### Step F — Start the app

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for Gradio to print a local URL. Open the exact URL shown in the terminal, normally:

```text
http://127.0.0.1:7860
```

To stop the server, focus the terminal and press **Ctrl+C**.

**You do not need to copy the old `data/`, `models/`, `outputs/`, or `.venv` folders from another PC to make a fresh install.** The app creates its working folders, and model weights are downloaded on demand. You do need internet access for the initial downloads.

---

## 4. Start the app

### Each time after setup

Open PowerShell and move into the folder where you cloned the repository. For the clone command above:

```powershell
cd $HOME\AI-Art-Style-Transfer
.\.venv\Scripts\python.exe app.py
```

If you chose another clone location, use that location instead.

### In VS Code

1. Open VS Code.
2. Choose **File → Open Folder**.
3. Select the cloned `AI-Art-Style-Transfer` folder.
4. Choose **Terminal → New Terminal**.
5. Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

The app opens in your browser. The terminal must stay running while you use it.

---

## 5. How model downloads work

The source-code repository intentionally does **not** store large model weights. On the first use of a style, the app may download the base diffusion model, ControlNet, and a style adapter from the model provider. The downloads are cached on that computer, so subsequent runs usually reuse them.

- Keep an internet connection for first-time downloads.
- Allow several GB of free space; different base models and adapters increase the total.
- A download can take a while. Do not close the terminal while a model is downloading.
- Some model repositories may require you to sign in, read/accept terms, or request access on the model page. Follow the instructions on that model's card. Never paste authentication tokens into code or screenshots.
- Downloaded model files are stored in local PyTorch/Hugging Face caches and/or the ignored `models/` directory. They are not automatically synchronized between PCs.
- For an 8 GB GPU, diffusion can be slow with CPU offloading. Close other GPU-heavy programs if generation runs out of memory.

---

## 6. Update the app later with `git pull`

Use this section when the project is already cloned on the computer and you want the latest code from GitHub. You do not need to clone it again every time.

First, go to the repository folder. Using the default clone location:

```powershell
cd $HOME\AI-Art-Style-Transfer
git status
```

If you have modified files you want to keep, commit them or back them up before updating. Then run:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

`git pull` updates version-controlled code and documentation. It does **not** download your old input photos or generated images from the other PC.

If `git pull` says the branch cannot be fast-forwarded, stop and inspect `git status` and `git log --oneline --decorate -5`. Do not force-push or delete local changes just to make the error disappear.

---

## 7. Use the app from a phone

The app runs on the **PC**. Your phone only opens the interface in its browser. Both devices must be connected to the **same trusted private Wi-Fi network**.

1. Start the app on your PC by opening PowerShell in the cloned project folder.
2. Set the host and port so the app is reachable from other devices on the local network:

```powershell
$env:AI_ART_STUDIO_HOST="0.0.0.0"
$env:AI_ART_STUDIO_PORT="7860"
.\.venv\Scripts\python.exe app.py
```

3. On the PC, open another PowerShell terminal and run:

```powershell
ipconfig
```

Find the **IPv4 Address** of the active Wi-Fi adapter, for example `192.168.1.23`.

4. If Windows Firewall prompts you, allow Python for **Private networks only**.
5. Connect your phone to the same Wi-Fi.
6. In the phone browser, enter the PC's address with port 7860. For example, if the PC IPv4 address was `192.168.1.23`, open:

```text
http://192.168.1.23:7860
```

Replace the example address with the actual IPv4 address from your PC. Keep the app terminal running while using it.

**Security:** local-network mode allows devices that can reach your PC on that network to access the interface. Use only a trusted private network. Do not configure router port forwarding or expose this unauthenticated interface to the public internet. To return to PC-only access, stop the app and restart it without `AI_ART_STUDIO_HOST`; it defaults to `127.0.0.1`.

---

## 8. Run on macOS or Linux

Git, Python, PyTorch, and compatible dependencies are required. Diffusion model execution is generally best supported on a suitable NVIDIA GPU; CPU mode will be slower.

Install Git and a compatible Python version using your OS's official package/download instructions. Then open Terminal:

```bash
git clone https://github.com/mohammednaeemog-sys/AI-Art-Style-Transfer.git
cd AI-Art-Style-Transfer
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
```

Install PyTorch using the official selector for your OS and hardware: https://pytorch.org/get-started/locally/ . Then install app dependencies and launch:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app.py
```

Open the local URL printed by Gradio. For phone access on the same private Wi-Fi, set environment variables before starting the app:

```bash
AI_ART_STUDIO_HOST=0.0.0.0 AI_ART_STUDIO_PORT=7860 ./.venv/bin/python app.py
```

Use your PC's local IP address on the phone, not `127.0.0.1`. Do not expose the app to an untrusted or public network.

---

## 9. Available styles and honest limitations

| Style | Model/method | What to expect |
|---|---|---|
| **Anime** | AnimeGANv2 feed-forward neural model | Fast photo-to-anime appearance; details may change. |
| **Starry Night (Neural)** | Dedicated feed-forward style model | Van Gogh-inspired colors/texture; a model-loader failure must be investigated rather than assumed successful. |
| **Candy, Mosaic, Rain Princess, Udnie (Neural)** | Published fast-neural-style checkpoints | Each is a fixed artistic style. |
| **Pop Art** | Stable Diffusion 1.5 + Canny ControlNet + Pop Art LoRA when compatible | Comic-book/printmaking-inspired stylization. |
| **Art Nouveau** | SDXL + SDXL Canny ControlNet + Mucha LoRA when compatible | Decorative botanical illustration; SDXL is heavier and slower. |
| **Ukiyo-e** | SD 1.5 + Canny ControlNet + Hiroshige LoRA when compatible | Japanese woodblock-print-inspired look. |
| **Impressionism** | SD 1.5 + Canny ControlNet + Monet LoRA when compatible | Painterly colors and brushwork. |
| **Cyberpunk** | Cyberpunk-focused checkpoint + Canny ControlNet, or an SD 1.5 prompt fallback | Neon/futuristic styling; checkpoint compatibility matters. |
| **Surrealism** | SD 1.5 + Canny ControlNet + style-specific prompt | Dreamlike appearance; exact semantic object placement is not guaranteed. |
| **Art Deco** | SD 1.5 + Canny ControlNet + prompt or compatible adapter | Geometric vintage-poster look. |
| **Watercolor** | SD 1.5 + Canny ControlNet + style prompt | Watercolor-inspired washes. |
| **Pencil Sketch** | SD 1.5 + Canny ControlNet + style prompt | Graphite/monochrome look. |
| **Oil Painting** | SD 1.5 + Canny ControlNet + oil-paint adapter when compatible | Painted surfaces and brush strokes. |
| **Pixel Art** | SD 1.5 + Canny ControlNet + PixelArtRedmond adapter when compatible | Pixel-art illustration; check its special license. |
| **Reference Style Transfer** | Experimental VGG19 optimization | Transfers texture/color statistics, but can create noisy output and does not recreate the reference composition semantically. |

The app uses **image-to-image + Canny ControlNet** for diffusion styles. Edge guidance helps preserve broad structure; it does not guarantee the exact same face, sign lettering, small objects, or number of details. If a LoRA cannot load, the app's status should tell you that it used a prompt + ControlNet fallback. A fallback is less style-specific than a successfully loaded adapter.

### General settings advice

- Start with art strength around `0.70–0.82`.
- Start with composition/edge preservation around `0.65–0.80`.
- Start with `32–36` diffusion steps.
- For a portrait, use lower strength (around `0.60–0.72`) and higher composition preservation (around `0.75–0.85`).
- Raise strength gradually if the effect looks too close to the original; lower it if faces or important objects change.
- More steps are not a guarantee of better style quality. Selecting a suitable, compatible style model matters more.

---

## 10. Local files, privacy, and GitHub publishing

### What stays local

- Source photographs uploaded into the app.
- Generated results saved in `outputs/`.
- Downloaded model weights and caches.
- The `.venv` Python environment.

The `.gitignore` is intended to keep private images, data, model weights, environments, and generated outputs out of future Git commits. Git ignores do not remove files that were already committed in repository history; inspect the GitHub repository if you are unsure whether a sensitive file was previously published.

### Publish one approved result

The app has a manual showcase publishing workflow:

1. Generate a result.
2. Inspect it and make sure you have permission to share it.
3. Tick the public-publishing approval checkbox.
4. Type `PUBLISH`.
5. Click **Publish approved image to GitHub**.

The intended workflow copies only that approved output into `examples/`, stages only that file, commits it, and pushes to `origin`. It does not upload the original input photo or the other files in `outputs/` through that button.

The repository is public, so a published showcase image may be copied by others. Do not publish private portraits or other images without consent.

---

## 11. Troubleshooting

### `py -3.12` is not found

Install Python 3.12 from https://www.python.org/downloads/windows/ and open a new PowerShell terminal. Run `py -0p` to see available Python installations.

### PowerShell blocks `.venv` activation

You can avoid activation and run the environment directly:

```powershell
.\.venv\Scripts\python.exe app.py
```

If you prefer activation, use a process-only policy (it applies to the current PowerShell process only), then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### `No module named ...`, or a dependency import error

Make sure the terminal is in the cloned project root, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Restart the app after installation. Do not randomly upgrade Diffusers, Transformers, PEFT and Hugging Face Hub independently.

### `CUDA available: False`

Install the correct PyTorch build using https://pytorch.org/get-started/locally/. A CPU build can run some functionality but diffusion is much slower.

### CUDA out of memory

Close other GPU-heavy applications, use a smaller input image, lower diffusion steps, or choose a lighter style model. SDXL is heavier than SD 1.5. If memory errors persist, try CPU mode understanding that it will be much slower.

### A model download fails or returns HTTP 401/403

Check internet access, inspect the terminal error, and visit the exact model card. Some model repositories may require account sign-in or acceptance of their terms. Retry after resolving the access issue. Do not post access tokens in screenshots or code.

### A style adapter fails

Read the on-screen status and terminal. If the app uses a prompt + ControlNet fallback, the output may be less distinctive. A fallback is not evidence that the intended LoRA was successfully loaded.

### The local page does not open

Keep the app terminal running. Open the exact URL printed there. If port 7860 is occupied, run:

```powershell
$env:AI_ART_STUDIO_PORT="7861"
python app.py
```

Then use the new URL printed by the app. For phone access, use the PC's IPv4 address and the correct port.

### `git clone` says the folder already exists

Choose a different parent folder or inspect the existing directory before deleting it. Do not delete local photos or outputs you want to keep.

### `git pull` fails

Run:

```powershell
git status
git remote -v
git branch --show-current
```

Back up or commit wanted local changes before pulling. Do not force-push to work around a rejected update without reviewing repository history.

---

## 12. Model sources and licenses

Models are downloaded from their publishers; their weights are not included in this source repository. The app may work locally for free after download, but the model licenses are not all identical. Review each license/model card before redistribution, public demonstrations, or commercial use.

- [Stable Diffusion 1.5 base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet Canny for SD 1.5](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)
- [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL Canny ControlNet](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)
- [Pop Art LoRA](https://huggingface.co/DesiAEye/DesiAEye-Pop-Art-LORA)
- [Alphonse Mucha / Art Nouveau LoRA](https://huggingface.co/KappaNeuro/alphonse-mucha-style)
- [Hiroshige / Ukiyo-e LoRA](https://huggingface.co/espressosession/latent-studio-hiroshige-lora)
- [Monet / Impressionism LoRA](https://huggingface.co/J-YOON/lora-monet-sd1.5)
- [Cyberpunk Anime Diffusion checkpoint](https://huggingface.co/AdamOswald1/Cyberpunk-Anime-Diffusion_with_support_for_Gen-Imp_characters)
- [Oil-painting LoRA](https://huggingface.co/Jehugging/oilpaint_lora)
- [PixelArtRedmond LoRA](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5)
- [Starry Night fast-style model](https://huggingface.co/Rohanify/Brawnz-StyleTransferSN)
- [PyTorch fast-neural-style examples](https://github.com/pytorch/examples/tree/main/fast_neural_style)

Model cards, file names, licensing, hosted access requirements, and Python package compatibility can change. The README is a starting point, not a permanent guarantee about third-party availability.
