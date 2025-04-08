import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import cv2
from dotenv import load_dotenv
import os
from utils_segment_image import (
    preprocess_image,
    get_nuclei_mask,
    get_map_white_pixels_to_respresentatives,
    calculate_and_save_histogram_and_return_R_cutoff,
    get_background_mask,
    create_overlay
)
from constants import (
    KERNEL_SIZE,
    DILATION_ITERATIONS,
    ALPHA_OVERLAY,
    RPB_THRESHOLD_PERCENTILE
)
from segmentation_service import process_image_segmentation_request, process_ndpi_image_segmentation_request
# Initialize app and logging
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost'])
logging.basicConfig(level=logging.INFO)


@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Single entry point for the /segment route.
    Wraps all logic in one function call that can invoke
    other helper functions as necessary.
    """
    try:
        file = request.files.get('image')
        if file is None:
            logging.error("No file received in request")
            return jsonify({"error": "No file provided"}), 400

        if file.filename.endswith('.ndpi'):
            
            return jsonify({"error": "NDPI file format not supported"}), 400
        if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
            logging.error("Unsupported file format")
            return jsonify({"error": "Unsupported file format"}), 400
        else:
            logging.info(f"Received file: {file.filename}")
            
            response_data = process_image_segmentation_request(file)

        logging.info(f"Segmentation successful, total cells: {response_data['total_cell_count']}")
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("An error occurred during segmentation")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_dotenv()
    backend_port = int(os.getenv('BACKEND_PORT'))
    app.run(debug=False, host='0.0.0.0', port=backend_port)
