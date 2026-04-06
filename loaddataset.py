# First, install the huggingface datasets library:
# pip install datasets Pillow

import os
import time
from glob import glob
from datasets import load_dataset
from PIL import Image

# 1. Create a folder for your training data
save_dir = "./webcam_training_data"
os.makedirs(save_dir, exist_ok=True)

hf_token = os.getenv("HF_TOKEN")
target_count = 100
max_consecutive_errors = 10

existing_files = sorted(glob(f"{save_dir}/ffhq_*.jpg"))
start_index = len(existing_files)

if start_index >= target_count:
    print(f"Already have {start_index}/{target_count} images in '{save_dir}'. Nothing to do.")
    raise SystemExit(0)

print("Downloading FFHQ Dataset...")
# 2. Load the FFHQ dataset from Hugging Face (streaming mode so we don't download all 90GB)
# We use a community mirror of FFHQ that works easily
def get_stream(skip_count: int):
    if hf_token:
        stream = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True, token=hf_token)
    else:
        print("HF_TOKEN not set. Continuing with unauthenticated download (lower rate limits).")
        stream = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)

    if skip_count > 0:
        stream = stream.skip(skip_count)
    return iter(stream)

print(f"Processing and saving {target_count} high-res training images...")
print(f"Found {start_index}/{target_count} existing images. Resuming from there...")

image_count = start_index
consecutive_errors = 0
dataset_iter = get_stream(skip_count=image_count)

# 3. Iterate through the dataset and save up to target_count images
while image_count < target_count:
    try:
        item = next(dataset_iter)
        img = item["image"]

        # Ensure it's in RGB mode (removes alpha channels if any exist)
        img = img.convert("RGB")

        # Resize to exactly 512x512 (Stable Diffusion 1.5's native resolution)
        # We use LANCZOS resampling to maintain maximum crispness during the resize
        img = img.resize((512, 512), Image.Resampling.LANCZOS)

        # Save the image
        img.save(f"{save_dir}/ffhq_{image_count:03d}.jpg", quality=100)

        image_count += 1
        consecutive_errors = 0
        if image_count % 10 == 0 or image_count == target_count:
            print(f"Saved {image_count}/{target_count} images...")
    except StopIteration:
        print("Reached end of dataset stream before target count.")
        break
    except Exception as error:
        consecutive_errors += 1
        print(
            f"Stream error while fetching image {image_count}: {error}. "
            f"Retry {consecutive_errors}/{max_consecutive_errors}..."
        )
        if consecutive_errors >= max_consecutive_errors:
            print("Too many consecutive errors. Stop and rerun to continue.")
            break
        time.sleep(2)
        dataset_iter = get_stream(skip_count=image_count)

print(f"Done! Your high-res training data is in '{save_dir}'")