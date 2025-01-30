import os
import logging
from scripts.extract_frames import extract_frames
from scripts.detect_damage import detect_damage
from scripts.estimate_volume import compute_volume
from scripts.generate_report import generate_pdf_report

# Configure logging
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "pipeline.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
VIDEO_PATH_EMPTY = "./video_datasets/empty_wagons.mp4"
VIDEO_PATH_FILLED = "./video_datasets/filled_wagons.mp4"
FRAME_OUTPUT_PATH_EMPTY = "./frames/empty_wagons/"
FRAME_OUTPUT_PATH_FILLED = "./frames/filled_wagons/"
DAMAGE_OUTPUT_PATH = "./damage_output/"
REPORT_PATH = "./reports/wagon_report.pdf"

# Ensure required directories exist
os.makedirs(FRAME_OUTPUT_PATH_EMPTY, exist_ok=True)
os.makedirs(FRAME_OUTPUT_PATH_FILLED, exist_ok=True)
os.makedirs(DAMAGE_OUTPUT_PATH, exist_ok=True)

def main():
    """Main function to execute the full pipeline."""
    logging.info("Starting Wagon Inspection System")

    # Step 1: Extract frames & count wagons
    logging.info("Extracting frames from empty wagon video")
    wagon_count_empty = extract_frames(VIDEO_PATH_EMPTY, FRAME_OUTPUT_PATH_EMPTY)
    
    logging.info("Extracting frames from filled wagon video")
    wagon_count_filled = extract_frames(VIDEO_PATH_FILLED, FRAME_OUTPUT_PATH_FILLED)
    
    logging.info(f"Empty Wagon Count: {wagon_count_empty}, Filled Wagon Count: {wagon_count_filled}")

    if wagon_count_empty != wagon_count_filled:
        logging.warning("Mismatch in empty and filled wagon count!")

    # Step 2: Process each wagon for damage detection and volume estimation
    damage_data = {}
    volume_data = {}

    logging.info("Starting damage detection and volume estimation")
    for wagon_id in range(1, min(wagon_count_empty, wagon_count_filled) + 1):
        empty_img = os.path.join(FRAME_OUTPUT_PATH_EMPTY, f"wagon_{wagon_id}_frame_10.jpg")
        filled_img = os.path.join(FRAME_OUTPUT_PATH_FILLED, f"wagon_{wagon_id}_frame_10.jpg")

        if not os.path.exists(empty_img) or not os.path.exists(filled_img):
            logging.warning(f"Missing images for Wagon {wagon_id}, skipping")
            continue

        # Detect damage
        damage_output_path = detect_damage(empty_img)
        damage_data[wagon_id] = damage_output_path
        logging.info(f"Damage detected for Wagon {wagon_id}, saved at {damage_output_path}")

        # Compute volume
        volume = compute_volume(empty_img, filled_img)
        volume_data[wagon_id] = volume
        logging.info(f"Estimated volume for Wagon {wagon_id}: {volume:.2f} cubic meters")

    # Step 3: Generate PDF Report
    logging.info("Generating final PDF report")
    generate_pdf_report(damage_data, volume_data, REPORT_PATH)

    logging.info("Pipeline execution complete!")

if __name__ == "__main__":
    main()
