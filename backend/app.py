from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64

import cv2

# StarDist imports (example for TensorFlow variant)
from stardist.models import StarDist2D
from stardist.plot import render_label

app = Flask(__name__)
CORS(app)  # Enable CORS if front-end is on a different port

# Load the StarDist model once (e.g., at startup).
# Example uses the publicly available "2D_versatile_fluo" model.
model = StarDist2D.from_pretrained("2D_versatile_fluo")

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Receive an image file, perform StarDist nucleus segmentation (YXC),
    and return the overlay (with outlines) in base64 format plus a count.
    """

    # 1. Get the file from the request
    file = request.files.get('image', None)
    if not file:
        return jsonify({"error": "No file found"}), 400

    # 2. Read the image via Pillow
    image_pil = Image.open(file)

    # 3. Convert to float32 (grayscale), shape = (Y, X)
    gray_pil = image_pil.convert('L')
    image_array = np.array(gray_pil, dtype=np.float32) / 255.0

    # 4. Expand dimensions to shape (Y, X, 1)
    #    Now StarDist sees Y (height), X (width), and C=1 channel
    image_array = np.expand_dims(image_array, axis=-1)  # shape = (Y, X, 1)

    # 5. Run StarDist with axes='YXC'
    labels, details = model.predict_instances(image_array, axes='YXC')
    print(details)

    # 6. Count how many nuclei
    num_nuclei = int(labels.max()) if labels is not None else 0

    # 7. Create an overlay for visualization
    #    Since 'image_array' is (Y, X, 1), replicate it into 3 channels for color display
    image_3c = np.repeat(image_array, 3, axis=-1)  # shape = (Y, X, 3)

    # Render label outlines (returns an RGB overlay)
    label_rgb = render_label(labels, img=image_3c)

    # 8. Convert overlay image (NumPy) back to PIL for base64 encoding
    overlay_pil = Image.fromarray((label_rgb * 255).astype(np.uint8))

    # 9. Encode as base64
    img_byte_array = io.BytesIO()
    overlay_pil.save(img_byte_array, format='PNG')
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

    # 10. Construct the response
    # Encode the original image as base64 for debugging
    original_img_byte_array = io.BytesIO()
    image_pil.save(original_img_byte_array, format='PNG')
    original_img_base64 = base64.b64encode(original_img_byte_array.getvalue()).decode('utf-8')

    response_data = {
        "segmented_image": img_base64,
        "nuclei_count": num_nuclei,
        #"segmented_image": original_img_base64,  # Include original image for debugging
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
