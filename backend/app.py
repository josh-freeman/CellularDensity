import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import cv2
from utils import (
    preprocess_image,
    get_mask,
    get_map_white_pixels_to_respresentatives,
    calculate_and_save_histogram_and_return_R_cutoff,
)

# Initialize app and logging
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost'])
logging.basicConfig(level=logging.INFO)

# Constants
KERNEL_SIZE = (3, 3)
DILATION_ITERATIONS = 2
RPB_THRESHOLD_PERCENTILE = 50
OVERLAY_COLOR = np.array([255, 255, 0], dtype=np.uint8)
ALPHA_OVERLAY = 0.5


def create_overlay(original_image, mask):
    overlay = original_image.copy()
    overlay[mask == 255] = ((1 - ALPHA_OVERLAY) * overlay[mask == 255] + ALPHA_OVERLAY * OVERLAY_COLOR).astype(np.uint8)
    return overlay


@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        file = request.files.get('image')
        if file is None:
            logging.error("No file received in request")
            return jsonify({"error": "No file provided"}), 400

        gray_array, image = preprocess_image(file)
        original_image = np.array(image)

        mask = get_mask(gray_array)
        kernel = np.ones(KERNEL_SIZE, np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)

        zones = get_map_white_pixels_to_respresentatives(dilated_mask)
        rpb_thresh = calculate_and_save_histogram_and_return_R_cutoff(RPB_THRESHOLD_PERCENTILE, original_image, zones)

        new_mask = np.zeros_like(mask)
        unfiltered_zones_count = 0

        for zone in zones:
            total_rpb = sum(original_image[x, y, 0] for x, y in zone)
            avg_rpb = total_rpb / len(zone)

            if avg_rpb <= rpb_thresh:
                unfiltered_zones_count += 1
                for x, y in zone:
                    new_mask[x, y] = 255

        overlay_image = create_overlay(original_image, new_mask)

        img_byte_arr = io.BytesIO()
        Image.fromarray(overlay_image).save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        response_data = {
            "total_cell_count": unfiltered_zones_count,
            "segmented_image": img_base64,
            "cell_type_count_table": []
        }

        logging.info(f"Segmentation successful, total cells: {unfiltered_zones_count}")
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("An error occurred during segmentation")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
