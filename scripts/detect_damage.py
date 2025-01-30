import cv2
import os
import torch
import yaml
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Load system configuration
CONFIG_PATH = "./config/system_config.yaml"
DETECTRON_CONFIG_PATH = "./config/detectron2.yaml"

with open(CONFIG_PATH, "r") as config_file:
    system_config = yaml.safe_load(config_file)

with open(DETECTRON_CONFIG_PATH, "r") as config_file:
    detectron_config = yaml.safe_load(config_file)

# Configure logging
LOG_DIR = system_config["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "detection.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load model configuration
MODEL_WEIGHTS = detectron_config["model"]["weights"]
MODEL_CONFIG = detectron_config["model"]["config_file"]
CONFIDENCE_THRESHOLD = detectron_config["roi_heads"]["score_thresh_test"]
OUTPUT_SCALE = detectron_config["visualization"]["output_scale"]

# Create output directory for damage detection results
DAMAGE_OUTPUT_PATH = system_config["damage_output_dir"]
os.makedirs(DAMAGE_OUTPUT_PATH, exist_ok=True)

# Load Detectron2 Model
cfg = get_cfg()
cfg.merge_from_file(f"detectron2/configs/{MODEL_CONFIG}")
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
cfg.MODEL.ROI_HEADS.NUM_CLASSES = detectron_config["roi_heads"]["num_classes"]
predictor = DefaultPredictor(cfg)

def detect_damage(image_path):
    """
    Detects damage in a wagon image using Mask R-CNN and saves output image.
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    outputs = predictor(img)

    # Visualize detection
    v = Visualizer(img[:, :, ::-1], scale=OUTPUT_SCALE)
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save output image
    output_filename = os.path.join(DAMAGE_OUTPUT_PATH, os.path.basename(image_path))
    cv2.imwrite(output_filename, vis.get_image()[:, :, ::-1])
    logging.info(f"Damage detected and saved: {output_filename}")

    return output_filename

# Batch processing for extracted wagon frames
if __name__ == "__main__":
    FRAME_PATH = system_config["frame_output_dir_empty"]
    image_files = [f for f in os.listdir(FRAME_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

    logging.info(f"Starting damage detection on {len(image_files)} images")
    
    for image_file in image_files:
        image_path = os.path.join(FRAME_PATH, image_file)
        detect_damage(image_path)

    logging.info("Damage detection process completed.")
