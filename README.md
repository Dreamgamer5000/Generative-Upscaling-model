# Generative-Upscaling-model

Domain-specific generative upscaling for degraded webcam portraits using:
- Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)
- ControlNet Tile (`lllyasviel/control_v11f1e_sd15_tile`)
- Custom LoRA trigger words: `upscale person`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Hugging Face Authentication (Recommended)

Set a token to avoid unauthenticated rate limits during model downloads:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Run the Gradio Webcam App

```bash
python app.py
```

The app opens a local Gradio URL in your browser where you can:
- Capture a webcam image or upload one from disk.
- Tune denoising strength.
- Optionally enable RealESRGAN post-upscaling and set output scale.
- Generate a restored output at the original webcam aspect ratio and exactly `2x` the original dimensions.

## Preprocessing Behavior

For better compatibility with the LoRA training distribution, the webcam frame is:
- Center-cropped to a square at `512x512`.
- Upscaled to `1024x1024` before inference.

## Optional RealESRGAN Post-Upscaling

The app includes an optional RealESRGAN post-process step after diffusion output.

- Enable `Apply RealESRGAN Post-Upscaling` in the UI.
- Set `RealESRGAN Output Scale` (for example, `2.0` for 2x final enlargement).

Regardless of internal diffusion/RealESRGAN processing size, the app returns the final image at exactly `original_width * 2` and `original_height * 2`.

On first use, the app downloads `RealESRGAN_x4plus.pth` into the local `weights/` directory.

## LoRA File

Place your LoRA file at:

`sdffhq_finetune.safetensors`

If your filename/path is different, update `LORA_WEIGHTS_PATH` in `app.py`.
