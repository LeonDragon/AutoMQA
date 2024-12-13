from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from imutils.perspective import four_point_transform
from helper.perspective_correction import adjust_perspective, adjust_perspective_crop_by_coordinates
from helper.est_answer_area import infer_answer_area_average_size
from gemini_utils import process_answer_key, process_student_answers
from helper.heic_converter import convert_single_fileBytes_to_img_obj, handle_uploaded_file
import io
import os
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'heic'}

def process_image(image_data, min_width=30, min_height=5, min_aspect_ratio=0.9, max_aspect_ratio=1.2):
    # Convert image data to numpy array
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Apply perspective correction
    img_np = adjust_perspective(img_np)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    
    # Find contours
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_coords = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if (w >= min_width and h >= min_height) and ar >= min_aspect_ratio and ar <= max_aspect_ratio:
            bubble_coords.append((x, y, w, h))
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Process answer area
    if bubble_coords:
        x_min, y_min, x_max, y_max = infer_answer_area_average_size(bubble_coords)
        if x_min is not None:
            padding = 15
            x_min -= padding + 90
            y_min -= padding
            x_max += padding
            y_max += padding

            answer_area_contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            warped = four_point_transform(img_np.copy(), answer_area_contour.reshape(4, 2))
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Split into columns
            warped_height, warped_width = warped.shape[:2]
            column_width = warped_width // 4
            columns = [
                warped[0:warped_height, 0:column_width],
                warped[0:warped_height, column_width:2 * column_width],
                warped[0:warped_height, 2 * column_width:3 * column_width],
                warped[0:warped_height, 3 * column_width:warped_width]
            ]
            
            header = img_np[0:y_min, 0:img_np.shape[1]]
            
            # Convert images to base64
            _, processed_img_encoded = cv2.imencode('.jpg', img_np)
            _, warped_encoded = cv2.imencode('.jpg', warped)
            column_encoded = []
            for col in columns:
                _, col_encoded = cv2.imencode('.jpg', col)
                column_encoded.append(base64.b64encode(col_encoded).decode())
            _, header_encoded = cv2.imencode('.jpg', header)
            
            return {
                'success': True,
                'processed_image': base64.b64encode(processed_img_encoded).decode(),
                'warped_image': base64.b64encode(warped_encoded).decode(),
                'columns': column_encoded,
                'header': base64.b64encode(header_encoded).decode()
            }
    
    return {'success': False, 'error': 'No bubbles found or processing failed'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_answer_key', methods=['POST'])
def handle_answer_key():
    if 'answer_key' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['answer_key']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Process the answer key
        image_data = file.read()
        img_array = np.frombuffer(image_data, np.uint8)
        answer_key_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # The image is already cropped by the frontend, so we can process it directly
        answer_keys = process_answer_key(answer_key_image)
        
        if answer_keys:
            return jsonify({'success': True, 'answer_keys': answer_keys})
        else:
            return jsonify({'error': 'Failed to extract answer key'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_student_sheet', methods=['POST'])
def handle_student_sheet():
    if 'student_sheet' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['student_sheet']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Get parameters from the request
        min_width = int(request.form.get('min_width', 30))
        min_height = int(request.form.get('min_height', 5))
        min_aspect_ratio = float(request.form.get('min_aspect_ratio', 0.9))
        max_aspect_ratio = float(request.form.get('max_aspect_ratio', 1.2))
        
        # Process the image
        image_data = file.read()
        result = process_image(image_data, min_width, min_height, min_aspect_ratio, max_aspect_ratio)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_with_gemini', methods=['POST'])
def handle_gemini_processing():
    try:
        data = request.json
        model_name = data.get('model_name', 'gemini-1.5-flash')
        columns = data.get('columns', [])
        answer_key_path = "answer_keys.json"
        
        # Convert base64 columns back to numpy arrays
        column_arrays = []
        for col_base64 in columns:
            col_data = base64.b64decode(col_base64)
            col_array = cv2.imdecode(np.frombuffer(col_data, np.uint8), cv2.IMREAD_COLOR)
            column_arrays.append(col_array)
        
        answers, scores = process_student_answers(column_arrays, model_name, answer_key_path)
        
        if answers is not None and scores is not None:
            return jsonify({
                'success': True,
                'answers': answers,
                'scores': {str(k): float(v) if v is not None else None for k, v in scores.items()}
            })
        else:
            return jsonify({'error': 'Failed to process answers'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_all_columns', methods=['POST'])
def handle_all_columns():
    print("\n=== Received request to /process_all_columns ===")
    try:
        data = request.json
        print("Request data received:", data is not None)
        
        model_name = data.get('model_name', 'gemini-1.5-flash')
        columns_base64 = data.get('columns', [])
        
        print("\n=== Request Details ===")
        print(f"Model name: {model_name}")
        print(f"Number of columns received: {len(columns_base64)}")
        
        if not columns_base64:
            print("Error: No columns provided in request")
            return jsonify({'error': 'No columns provided'}), 400
            
        # Convert base64 columns back to numpy arrays
        column_arrays = []
        for i, col_base64 in enumerate(columns_base64):
            try:
                print(f"\nProcessing Column {i+1}:")
                col_data = base64.b64decode(col_base64)
                print(f"- Base64 decoded successfully")
                
                col_array = cv2.imdecode(np.frombuffer(col_data, np.uint8), cv2.IMREAD_COLOR)
                if col_array is not None:
                    print(f"- Converted to numpy array: shape={col_array.shape}, dtype={col_array.dtype}")
                    column_arrays.append(col_array)
                else:
                    print(f"Error: Failed to decode column {i+1}")
                    raise ValueError(f"Failed to decode column {i+1}")
            except Exception as e:
                print(f"Error processing column {i+1}: {str(e)}")
                raise
        
        print(f"\n=== Column Processing Complete ===")
        print(f"Successfully processed {len(column_arrays)} columns")
        
        # Process all columns with Gemini
        print("\n=== Starting Gemini Processing ===")
        all_answers, scores = process_student_answers(column_arrays, model_name, 'answer_keys.json')
        
        print("\n=== Gemini Processing Complete ===")
        print(f"Answers received: {all_answers is not None}")
        print(f"Scores received: {scores is not None}")
        
        if all_answers is None:
            raise ValueError("Failed to get answers from Gemini")
            
        # Split answers into groups of 15
        column_results = [all_answers[i:i+15] for i in range(0, len(all_answers), 15)]
        
        print("\n=== Preparing Response ===")
        print(f"Number of column results: {len(column_results)}")
        print(f"Scores: {scores}")
        
        return jsonify({
            'success': True,
            'column_results': column_results,
            'scores': scores
        })
        
    except Exception as e:
        print(f"\n=== Error in /process_all_columns ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)