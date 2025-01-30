import cv2
import os
import yaml
import logging

# Load system configuration
CONFIG_PATH = "./config/system_config.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

# Configure logging
LOG_DIR = config["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "extraction.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Motion detection parameters
MOTION_THRESHOLD = config["frame_extraction"]["motion_detection_threshold"]
FRAME_INTERVAL = config["frame_extraction"]["frame_interval"]

# Frame storage directories
FRAME_OUTPUT_PATH_EMPTY = config["frame_output_dir_empty"]
FRAME_OUTPUT_PATH_FILLED = config["frame_output_dir_filled"]
os.makedirs(FRAME_OUTPUT_PATH_EMPTY, exist_ok=True)
os.makedirs(FRAME_OUTPUT_PATH_FILLED, exist_ok=True)

# Background subtraction model for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

def extract_frames(video_path, output_path):
    """
    Extracts frames from video where motion is detected and assigns wagon IDs.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    wagon_count = 0
    previous_wagon_detected = False

    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Detect contours (wagon detection)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wagon_detected = any(cv2.contourArea(cnt) > MOTION_THRESHOLD for cnt in contours)

        if wagon_detected:
            if not previous_wagon_detected:
                wagon_count += 1  # Increment count when a new wagon appears

            if frame_count % FRAME_INTERVAL == 0:
                frame_filename = os.path.join(output_path, f"wagon_{wagon_count}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                logging.info(f"Saved frame for Wagon {wagon_count}: {frame_filename}")

        previous_wagon_detected = wagon_detected
        frame_count += 1

    cap.release()
    logging.info(f"Extracted frames from {video_path} -> {wagon_count} wagons counted")
    return wagon_count

# Execute extraction for both empty and filled wagon videos
if __name__ == "__main__":
    logging.info("Starting frame extraction process")

    wagon_count_empty = extract_frames(config["video_paths"]["empty_wagons"], FRAME_OUTPUT_PATH_EMPTY)
    wagon_count_filled = extract_frames(config["video_paths"]["filled_wagons"], FRAME_OUTPUT_PATH_FILLED)

    logging.info(f"Final Wagon Count - Empty: {wagon_count_empty}, Filled: {wagon_count_filled}")

    if wagon_count_empty != wagon_count_filled:
        logging.warning("Mismatch in empty and filled wagon counts!")
