# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install diffusers transformers accelerate safetensors gradio pillow

"""Gradio webcam inference app for domain-specific generative super-resolution.

This script deploys a Stable Diffusion 1.5 + ControlNet Tile + LoRA pipeline
for enhancing degraded webcam portraits in an academic demonstration setting.
"""

from __future__ import annotations

import sys
import types
import gc
from typing import Any, cast
from pathlib import Path

import torch
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageOps
from gradio_imageslider import ImageSlider
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1e_sd15_tile"
LORA_WEIGHTS_PATH = "sdffhq_finetune.safetensors"
# Change this file name/path if your detail LoRA safetensors file is different.
DETAIL_LORA_PATH = "add_detail.safetensors"
DETAIL_LORA_WEIGHT = 1.0
TRAIN_RESOLUTION = 512
MODEL_CANVAS_SIZE = 1024
REALESRGAN_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
REALESRGAN_WEIGHTS_DIR = "weights"
REALESRGAN_TILE_SIZE = 256
SD_TILE_SIZE = 512
SD_TILE_OVERLAP = 64
SD_NUM_INFERENCE_STEPS = 24
SD_GUIDANCE_SCALE = 6.0
SD_UPSCALE_FACTOR = 1.5

POSITIVE_PROMPT = (
    "upscale person macro photography, highly detailed pores, intricate skin texture, peach fuzz, "
    "sharp focus, 8k uhd, dslr, soft studio lighting, masterpiece, ultra-detailed, "
    "photorealistic face, fine fabric weave"
)

NEGATIVE_PROMPT = (
    "smooth, plastic, cartoon, blurry, out of focus, overexposed, washed out, "
    "cgi, 3d render, illustration, bad art, deformed, bad anatomy, bad lighting, "
    "jpeg artifacts, noise"
)

def build_pipeline() -> StableDiffusionControlNetImg2ImgPipeline:
    """Construct and configure the SD1.5 ControlNet Img2Img pipeline.

    Architecture requirements implemented:
    - Base model: runwayml/stable-diffusion-v1-5 in fp16.
    - ControlNet: lllyasviel/control_v11f1e_sd15_tile in fp16.
    - Scheduler: DDIMScheduler.
    - Memory optimization: CPU offload for safer execution on low-VRAM GPUs.
    - LoRA adapter: custom domain-specific upscaler adapter.
    """
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # Disable the default safety checker to avoid false-positive black outputs
    # during domain-specific portrait restoration experiments.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # Change this file name/path if your LoRA safetensors file is different.
    pipe.load_lora_weights(LORA_WEIGHTS_PATH, adapter_name="upscaler")
    pipe.load_lora_weights(DETAIL_LORA_PATH, adapter_name="detailer")
    pipe.set_adapters(["upscaler", "detailer"], adapter_weights=[0.8, DETAIL_LORA_WEIGHT])

    pipe.enable_model_cpu_offload()
    return pipe


PIPELINE = build_pipeline()
REALESRGAN_UPSAMPLERS: dict[str, Any] = {}


def ensure_torchvision_compat() -> None:
    """Patch torchvision import path expected by older BasicSR releases."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        import torchvision.transforms.functional as functional

        compat_module = types.ModuleType("torchvision.transforms.functional_tensor")
        setattr(compat_module, "rgb_to_grayscale", functional.rgb_to_grayscale)
        sys.modules["torchvision.transforms.functional_tensor"] = compat_module


def get_realesrgan_upsampler(use_gpu: bool = True) -> Any:
    """Lazily create and return a RealESRGAN upsampler instance."""
    device_key = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    if device_key in REALESRGAN_UPSAMPLERS:
        return REALESRGAN_UPSAMPLERS[device_key]

    ensure_torchvision_compat()

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    weights_path = load_file_from_url(
        url=REALESRGAN_WEIGHTS_URL,
        model_dir=REALESRGAN_WEIGHTS_DIR,
        progress=True,
        file_name="RealESRGAN_x4plus.pth",
    )

    if not Path(weights_path).exists():
        raise RuntimeError("RealESRGAN weights download failed.")

    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=REALESRGAN_TILE_SIZE,
        tile_pad=10,
        pre_pad=0,
        half=device_key == "cuda",
        gpu_id=0 if device_key == "cuda" else None,
    )
    REALESRGAN_UPSAMPLERS[device_key] = upsampler
    return upsampler


def clear_cuda_cache() -> None:
    """Release unreferenced CUDA memory to reduce OOM risk."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def apply_realesrgan(
    image: Image.Image,
    outscale: float,
    use_gpu: bool = True,
) -> Image.Image:
    """Apply RealESRGAN enhancement on a PIL image."""
    upsampler = get_realesrgan_upsampler(use_gpu=use_gpu)

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    output_bgr, _ = upsampler.enhance(image_bgr, outscale=outscale)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


def get_tile_weight(
    tile_width: int,
    tile_height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    image_width: int,
    image_height: int,
    overlap: int,
) -> np.ndarray:
    """Create a soft blending mask for tiled stitching."""
    weight = np.ones((tile_height, tile_width), dtype=np.float32)
    feather = min(overlap, tile_width // 2, tile_height // 2)

    if feather <= 0:
        return weight

    ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)

    if left > 0:
        weight[:, :feather] *= ramp
    if right < image_width:
        weight[:, -feather:] *= ramp[::-1]
    if top > 0:
        weight[:feather, :] *= ramp[:, None]
    if bottom < image_height:
        weight[-feather:, :] *= ramp[::-1, None]

    return weight


def run_tiled_sd_refinement(image: Image.Image, strength: float) -> Image.Image:
    """Run overlap-tiled SD refinement and blend tiles seamlessly."""
    image_rgb = image.convert("RGB")
    image_width, image_height = image_rgb.size
    tile_size = SD_TILE_SIZE
    overlap = SD_TILE_OVERLAP
    stride = max(1, tile_size - overlap)

    image_np = np.array(image_rgb, dtype=np.float32)
    accum = np.zeros_like(image_np, dtype=np.float32)
    weights = np.zeros((image_height, image_width), dtype=np.float32)

    top_positions = list(range(0, image_height, stride))
    left_positions = list(range(0, image_width, stride))

    if top_positions and top_positions[-1] != max(0, image_height - tile_size):
        top_positions.append(max(0, image_height - tile_size))
    if left_positions and left_positions[-1] != max(0, image_width - tile_size):
        left_positions.append(max(0, image_width - tile_size))

    top_positions = sorted(set(top_positions))
    left_positions = sorted(set(left_positions))

    for top in top_positions:
        for left in left_positions:
            right = min(left + tile_size, image_width)
            bottom = min(top + tile_size, image_height)

            tile = image_rgb.crop((left, top, right, bottom))
            tile_width = right - left
            tile_height = bottom - top

            if tile.size != (tile_size, tile_size):
                padded_tile = Image.new("RGB", (tile_size, tile_size))
                padded_tile.paste(tile, (0, 0))
                tile_for_pipeline = padded_tile
            else:
                tile_for_pipeline = tile

            output = cast(
                Any,
                PIPELINE(
                    prompt=POSITIVE_PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=tile_for_pipeline,
                    control_image=tile_for_pipeline,
                    strength=strength,
                    num_inference_steps=SD_NUM_INFERENCE_STEPS,
                    guidance_scale=SD_GUIDANCE_SCALE,
                    controlnet_conditioning_scale=1.0,
                    output_type="pil",
                    return_dict=True,
                ),
            )

            enhanced_tile = cast(Image.Image, output.images[0]).crop(
                (0, 0, tile_width, tile_height)
            )
            enhanced_np = np.array(enhanced_tile, dtype=np.float32)

            tile_weight = get_tile_weight(
                tile_width=tile_width,
                tile_height=tile_height,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                image_width=image_width,
                image_height=image_height,
                overlap=overlap,
            )

            accum[top:bottom, left:right] += enhanced_np * tile_weight[:, :, None]
            weights[top:bottom, left:right] += tile_weight

            clear_cuda_cache()

    weights = np.maximum(weights, 1e-6)
    stitched = accum / weights[:, :, None]
    stitched = np.clip(stitched, 0, 255).astype(np.uint8)
    return Image.fromarray(stitched)


def generate_upscaled_image(
    webcam_image: Image.Image,
    strength: float,
    enable_realesrgan: bool,
    realesrgan_outscale: float,
) -> tuple[Image.Image, tuple[Image.Image, Image.Image]]:
    """Run super-resolution inference from webcam input.

    Args:
        webcam_image: Input PIL image captured from the laptop webcam.
        strength: Denoising strength from the Gradio slider.
        enable_realesrgan: Whether to run RealESRGAN as a post-process step.
        realesrgan_outscale: RealESRGAN post-upscale factor.

    Returns:
        A stitched full-image output and a before-after slider pair.
    """
    if webcam_image is None:
        raise gr.Error("Please capture a webcam photo before running enhancement.")

    source_image = webcam_image.convert("RGB")
    source_width, source_height = source_image.size

    upscale_canvas = source_image.resize(
        (
            max(64, int(source_width * SD_UPSCALE_FACTOR)),
            max(64, int(source_height * SD_UPSCALE_FACTOR)),
        ),
        Image.Resampling.BICUBIC,
    )

    stitched_output = run_tiled_sd_refinement(upscale_canvas, strength=strength)

    clear_cuda_cache()

    if not enable_realesrgan:
        return stitched_output, (upscale_canvas, stitched_output)

    try:
        realesrgan_output = apply_realesrgan(
            stitched_output,
            outscale=realesrgan_outscale,
            use_gpu=True,
        )
        previous_image = upscale_canvas
        if previous_image.size != realesrgan_output.size:
            previous_image = previous_image.resize(
                realesrgan_output.size,
                Image.Resampling.BICUBIC,
            )
        return stitched_output, (previous_image, realesrgan_output)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise gr.Error(f"RealESRGAN failed: {error}") from error

        clear_cuda_cache()
        try:
            realesrgan_output = apply_realesrgan(
                stitched_output,
                outscale=realesrgan_outscale,
                use_gpu=False,
            )
            previous_image = upscale_canvas
            if previous_image.size != realesrgan_output.size:
                previous_image = previous_image.resize(
                    realesrgan_output.size,
                    Image.Resampling.BICUBIC,
                )
            return stitched_output, (previous_image, realesrgan_output)
        except Exception as cpu_error:
            raise gr.Error(f"RealESRGAN fallback failed: {cpu_error}") from cpu_error
    except Exception as error:
        raise gr.Error(f"RealESRGAN failed: {error}") from error


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 Domain-Specific Generative Super-Resolution")
    gr.Markdown(
        "This interface runs an Ultimate SD Upscale-style overlap tiled refinement "
        "with custom Stable Diffusion 1.5 LoRA + ControlNet Tile conditioning."
    )

    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(
                sources=["webcam", "upload"],
                type="pil",
                label="Capture Webcam Photo or Upload Image",
            )
            denoise_strength = gr.Slider(
                minimum=0.1,
                maximum=0.7,
                value=0.40,
                step=0.01,
                label="Denoising Strength",
            )
            enable_realesrgan = gr.Checkbox(
                value=True,
                label="Apply RealESRGAN Post-Upscaling",
            )
            realesrgan_outscale = gr.Slider(
                minimum=1.0,
                maximum=4.0,
                value=2.0,
                step=0.5,
                label="RealESRGAN Output Scale",
            )
            enhance_button = gr.Button("Enhance Image", variant="primary")

        with gr.Column():
            cropped_output = gr.Image(
                type="pil",
                label="Stitched Ultimate SD Output",
            )
            output_image = ImageSlider(
                label="Previous vs New",
                type="pil",
            )

    enhance_button.click(
        fn=generate_upscaled_image,
        inputs=[
            webcam_input,
            denoise_strength,
            enable_realesrgan,
            realesrgan_outscale,
        ],
        outputs=[cropped_output, output_image],
    )


if __name__ == "__main__":
    app.launch()
