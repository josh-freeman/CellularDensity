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
import tempfile
import os
import json

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

    # Create a temporary directory
    temp_dir_in = tempfile.mkdtemp()
    temp_dir_out = tempfile.mkdtemp()
    NR_TYPES = 6
    type_info_path = 'type_info.json'
    infer_command = f"""python run_infer.py \
            --gpu='0,1,2' \
            --nr_types={NR_TYPES} \
            --type_info_path={type_info_path} \
            --batch_size=64 \
            --model_mode=fast \
            --model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
            --nr_inference_workers=8 \
            --nr_post_proc_workers=16 \
            tile \
            --input_dir={temp_dir_in} \
            --output_dir={temp_dir_out} \
            --mem_usage=0.6 \
            --draw_dot 
    """
    # Save the image to the input directory
    file.save(os.path.join(temp_dir_in, file.filename))
    
    # run the inference from models/hover_net
    os.chdir('models/hover_net')
    os.system(infer_command)
    os.chdir('../..')

    # get the image overlay from temp_dir_out/overlay (replace the extension with .png). 
    name_with_extension = file.filename.split('.')
    overlay_path = os.path.join(temp_dir_out, 'overlay', name_with_extension[0] + '.png')
    overlay_img = Image.open(overlay_path)
    img_byte_array = io.BytesIO()
    overlay_img.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')
    # get cell_type_count_table from temp_dir_out/json (replace the extension with .json)
    json_path = os.path.join(temp_dir_out, 'json', name_with_extension[0] + '.json')
    
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    cell_type_count_table = [{"count":0} for _ in range(NR_TYPES)]
    for i in range(1, len(data["nuc"]) + 1):
        cell_type = data["nuc"][str(i)]["type"]
        cell_type_count_table[int(cell_type)] = {"count": cell_type_count_table[int(cell_type)]["count"] + 1}
    
    # Open the type info file and read its contents
    with open(os.path.join('models/hover_net', type_info_path), 'r') as type_info_file:
        type_info = json.load(type_info_file)
        for key, value in enumerate(cell_type_count_table):
            if value:
                cell_type_count_table[key]['name'] = type_info[str(key)][0]
                cell_type_count_table[key]['color'] = type_info[str(key)][1]
    
    # 10. Return the base64 string in a JSON response
    return jsonify({"segmented_image": img_base64, "cell_type_count_table": cell_type_count_table})

if __name__ == '__main__':
    # Typically in dev you'd run: flask run
    # but for demonstration:
    app.run(debug=True, host='0.0.0.0', port=8080)
