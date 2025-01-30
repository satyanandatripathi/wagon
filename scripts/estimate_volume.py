import cv2
import os
import torch
import json
import logging
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Load system configuration
CONFIG_PATH = "./config/system_config.yaml"
MIDAS_CONFIG_PATH = "./config/midas_config.json"

with open(CONFIG_PATH, "r") as config_file:
    system_config = json.load(config_file)

with open(MIDAS_CONFIG_PATH, "r") as config_file:
    midas_config = json.load(config_file)

# Configure logging
LOG_DIR = system_config["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "volume_estimation.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load MiDas Model
MODEL_WEIGHTS = midas_config["model"]["weights"]
DEVICE = midas_config["model"]["device"]
PIXEL_AREA = midas_config["depth_processing"]["pixel_area"]

midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(DEVICE)
midas_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

def estimate_depth(image_path):
    """
    Estimate depth from a single wagon image using MiDas (DPT).
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = midas_processor(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        depth_map = midas_model(**inputs).predicted_depth

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    return depth_map

def compute_volume(empty_image, filled_image):
    """
    Compute the material volume by comparing depth maps of empty and filled wagons.
    """
    empty_depth = estimate_depth(empty_image)
    filled_depth = estimate_depth(filled_image)

    if empty_depth is None or filled_depth is None:
        logging.error(f"Depth estimation failed for {empty_image} or {filled_image}")
        return 0

    # Compute depth difference (material thickness)
    depth_diff = filled_depth.astype(np.float32) - empty_depth.astype(np.float32)
    depth_diff[depth_diff < 0] = 0  # Remove negative values

    # Estimate volume (Sum of all depth pixels multiplied by pixel area)
    volume = np.sum(depth_diff) * PIXEL_AREA
    logging.info(f"Computed volume: {volume:.2f} cubic meters")

    return volume

# Batch processing for all extracted frames
if __name__ == "__main__":
    FRAME_PATH_EMPTY = system_config["frame_output_dir_empty"]
    FRAME_PATH_FILLED = system_config["frame_output_dir_filled"]

    image_files_empty = sorted([f for f in os.listdir(FRAME_PATH_EMPTY) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_files_filled = sorted([f for f in os.listdir(FRAME_PATH_FILLED) if f.endswith(('.png', '.jpg', '.jpeg'))])

    volume_data = {}

    logging.info("Starting volume estimation process")

    for i, (empty_img, filled_img) in enumerate(zip(image_files_empty, image_files_filled), start=1):
        empty_path = os.path.join(FRAME_PATH_EMPTY, empty_img)
        filled_path = os.path.join(FRAME_PATH_FILLED, filled_img)

        if not os.path.exists(empty_path) or not os.path.exists(filled_path):
            logging.warning(f"Missing images for Wagon {i}, skipping volume estimation")
            continue

        volume_data[i] = compute_volume(empty_path, filled_path)
        logging.info(f"Wagon {i}: Estimated volume = {volume_data[i]:.2f} cubic meters")

    logging.info("Volume estimation process completed.")
