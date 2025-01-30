import os
import json
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Load system configuration
CONFIG_PATH = "./config/system_config.yaml"

with open(CONFIG_PATH, "r") as config_file:
    system_config = json.load(config_file)

# Configure logging
LOG_DIR = system_config["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "report_generation.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Report settings
REPORT_PATH = os.path.join(system_config["report_dir"], system_config["report_filename"])
DAMAGE_OUTPUT_PATH = system_config["damage_output_dir"]

def generate_pdf_report(damage_data, volume_data, report_path=REPORT_PATH):
    """
    Generates a structured PDF report summarizing damage detection and volume estimation results.
    """
    logging.info("Generating PDF report")

    # Create a new PDF canvas
    c = canvas.Canvas(report_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, 750, "Wagon Damage & Volume Report")
    
    # Section: Damage Detection
    y_position = 700
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, "Damage Detection Results")
    y_position -= 30

    for wagon_id, damage_img in damage_data.items():
        if y_position < 100:  # Create a new page if space runs out
            c.showPage()
            y_position = 750

        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, f"Wagon {wagon_id}: Damage Detected")
        y_position -= 20

        if os.path.exists(damage_img):
            img_reader = ImageReader(damage_img)
            c.drawImage(img_reader, 100, y_position - 120, width=200, height=100)
            y_position -= 140
        else:
            c.drawString(100, y_position, "No Image Available")
            y_position -= 20

    # Section: Volume Estimation
    y_position -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, "Material Volume Estimations")
    y_position -= 30

    for wagon_id, volume in volume_data.items():
        if y_position < 100:  # New page if required
            c.showPage()
            y_position = 750

        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, f"Wagon {wagon_id}: {volume:.2f} cubic meters")
        y_position -= 30

    # Save PDF
    c.save()
    logging.info(f"Report successfully generated: {report_path}")
    print(f"PDF Report generated: {report_path}")

# Example Usage
if __name__ == "__main__":
    # Simulated test data (replace with actual data from `main.py`)
    sample_damage_data = {
        1: "./damage_output/wagon_1_damage.jpg",
        2: "./damage_output/wagon_2_damage.jpg"
    }

    sample_volume_data = {
        1: 25.7,
        2: 30.2
    }

    generate_pdf_report(sample_damage_data, sample_volume_data)
