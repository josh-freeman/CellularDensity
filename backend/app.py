from disjoint_set import DisjointSet
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
from utils import preprocess_image, get_mask, get_map_white_pixels_to_respresentatives, calculate_and_save_histogram_and_return_R_cutoff

app = Flask(__name__)
CORS(app)

import cv2

@app.route('/segment', methods=['POST'])
def segment_image():
    # 1. Get the file from the request
    file = request.files.get('image', None)
    if not file:
        return jsonify({"error": "No file found"}), 400
    
    gray_array, image = preprocess_image(file)
    mask = get_mask(gray_array)
    img = Image.fromarray(mask)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    zones = get_map_white_pixels_to_respresentatives(mask) # list of lists of (x, y) coordinates
    original_image = np.array(image)
    
    rpb_thresh = calculate_and_save_histogram_and_return_R_cutoff(50, original_image, zones)

    # make a new mask with only the pixels that have R+B > rpb_thresh
    new_mask = np.zeros_like(mask)
    unfiltered_zones_count = 0
    for zone in zones:
        total_rpb = 0
        for x, y in zone:
            total_rpb += np.int64(original_image[x, y, 0])
        avg_rpb = total_rpb // len(zone)

        if avg_rpb <= rpb_thresh:
            unfiltered_zones_count += 1
            for x, y in zone:
                new_mask[x, y] = 255



    img = Image.fromarray(new_mask)
    # Overlay the white parts of the mask on the original image with a transparent alpha in yellow
    overlay = np.array(original_image, dtype=np.uint8)
    yellow = [255, 255, 0]  # RGB for yellow
    alpha = 0.5  # Transparency level

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if new_mask[x, y] == 255:  # If the pixel is white in the mask
                overlay[x, y] = (1 - alpha) * overlay[x, y] + alpha * np.array(yellow)

    img = Image.fromarray(overlay.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    response_data = {
        "total_cell_count": unfiltered_zones_count,
        "segmented_image": img_base64,
        "cell_type_count_table": []
    }
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
