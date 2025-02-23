from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS if React front-end is served on a different port.

import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from skimage.measure import label, regionprops

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Receive an image file, perform a GPU-accelerated threshold segmentation,
    and return the image with nuclei circled in base64 format.
    """

    # 1. Get the file from the request
    file = request.files.get('image', None)
    if not file:
        return jsonify({"error": "No file found"}), 400

    # 2. Load the image (PIL)
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Convert to numpy array (H,W,3)
    img_np = np.array(img)

    # 3. Convert to grayscale
    gray = img.convert('L')
    gray_np = np.array(gray)

    # 4. Move the grayscale image to the GPU as a PyTorch tensor
    #    shape: (1, 1, H, W) so we can do operations easily
    gray_tensor = torch.from_numpy(gray_np).float().unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        gray_tensor = gray_tensor.cuda()

    # Example threshold (static or Otsu). Let's do a simple static threshold of 128
    # For something more dynamic, you could implement Otsu or another method on GPU as well.
    threshold_value = 128.0
    # Binarize on GPU
    # result is 1 where gray > threshold, else 0
    binary_tensor = (gray_tensor > threshold_value).float()

    # 5. Convert the GPU result back to CPU numpy array, shape (H, W)
    binary_mask = binary_tensor.squeeze().detach().cpu().numpy().astype(np.uint8)

    # Optionally scale mask to 0/255 for visualization as a black/white image
    binary_mask_255 = binary_mask * 255

    # 6. Identify connected components (nuclei) with scikit-image
    #    Each connected component (nucleus) will get its own label
    labeled_mask = label(binary_mask, connectivity=2)  # connectivity=2 -> 8-connected

    # 7. Measure properties of each labeled region (like centroid, area, bounding boxes, etc.)
    props = regionprops(labeled_mask)

    # 8. Draw circles around each detected nucleus on the original color image
    #    We'll convert the original image to BGR format for OpenCV drawing
    circled_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for region in props:
        # region.centroid -> (y, x)
        y, x = region.centroid
        # radius -> half of equivalent diameter (or you can use bounding box size)
        radius = int(region.equivalent_diameter / 2)

        # Draw a circle in green color
        # Make sure you cast x,y to int
        cv2.circle(circled_img, (int(x), int(y)), radius, (0, 255, 0), 2)

    # Convert back to RGB for Pillow
    circled_img_rgb = cv2.cvtColor(circled_img, cv2.COLOR_BGR2RGB)
    circled_pil = Image.fromarray(circled_img_rgb)

    # 9. Encode the circled image as base64 string
    img_buffer = io.BytesIO()
    circled_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    # 10. Return the base64 string in a JSON response
    return jsonify({"segmented_image": img_base64})

if __name__ == '__main__':
    # Typically in dev you'd run: flask run
    # but for demonstration:
    app.run(debug=True, host='0.0.0.0', port=8080)
