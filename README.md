# AutoMQA: Automated Multiple-Choice Assessment

AutoMQA is a Flask-based web application that automates the grading of multiple-choice assessments using computer vision and AI. It processes scanned answer sheets, detects and analyzes answer bubbles, and evaluates responses using Gemini AI.

## Features

* **Automated Grading:** Eliminates manual grading of multiple-choice tests
* **Image Processing:** Uses OpenCV for perspective correction, bubble detection, and image enhancement
* **AI-Powered Evaluation:** Leverages Gemini AI for accurate answer evaluation
* **Web Interface:** Provides REST API endpoints for easy integration
* **File Format Support:** Handles JPG, PNG, and HEIC formats
* **Detailed Reporting:** Provides processed images, detected answers, and scoring breakdown

![Application Interface](https://github.com/user-attachments/assets/37699322-d593-4584-8b7c-ca767fdef17c)

![image](https://github.com/user-attachments/assets/d39086b6-c748-4b7d-ae6c-d5672ca058c5)



*Figure 1: Main application interface showing image upload and processing controls*

## Technology Stack

* **Python 3**
* **Flask** - Web framework
* **OpenCV** - Image processing and computer vision
* **Gemini AI** - Answer evaluation and scoring
* **NumPy** - Numerical computations
* **Pillow** - Image manipulation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/AutoMQA.git
cd AutoMQA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export FLASK_APP=flask_app.py
export FLASK_ENV=development
```

4. Run the application:
```bash
flask run
```

## API Endpoints

### POST /process_answer_key
Processes the answer key image

**Request:**
- Multipart form with 'answer_key' file (JPG/PNG/HEIC)

**Response:**
```json
{
  "success": true,
  "answer_keys": {
    "1": "A",
    "2": "B",
    ...
  }
}
```

### POST /process_student_sheet
Processes a student answer sheet

**Request:**
- Multipart form with 'student_sheet' file (JPG/PNG/HEIC)
- Form parameters:
  - min_width: Minimum bubble width (default: 20)
  - min_height: Minimum bubble height (default: 4)
  - min_aspect_ratio: Minimum bubble aspect ratio (default: 0.7)
  - max_aspect_ratio: Maximum bubble aspect ratio (default: 1.4)

**Response:**
```json
{
  "success": true,
  "processed_image": "base64 encoded image",
  "warped_image": "base64 encoded image",
  "columns": ["base64 encoded column 1", ...],
  "header": "base64 encoded header"
}
```

### POST /process_with_gemini
Processes answer columns with Gemini AI

**Request:**
```json
{
  "model_name": "gemini-1.5-flash",
  "columns": ["base64 encoded column 1", ...]
}
```

**Response:**
```json
{
  "success": true,
  "answers": ["A", "B", ...],
  "scores": {
    "1": 1.0,
    "2": 0.0,
    ...
  },
  "all_responses": [
    "Gemini response 1",
    ...
  ]
}
```

## File Requirements

* Supported formats: JPG, PNG, HEIC
* Maximum file size: 16MB
* Answer sheets should have:
  - Clear bubble markings
  - Minimal skew or distortion
  - Good contrast between bubbles and background

## Example Usage

1. Start the Flask server:
```bash
flask run
```

2. Process an answer key:
```bash
curl -X POST -F "answer_key=@answer_key.jpg" http://localhost:5000/process_answer_key
```

3. Process a student sheet:
```bash
curl -X POST -F "student_sheet=@student_sheet.jpg" http://localhost:5000/process_student_sheet
```

4. Evaluate answers with Gemini:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "model_name": "gemini-1.5-flash",
  "columns": ["base64_column_1", ...]
}' http://localhost:5000/process_with_gemini
```

![[Processing Results](https://github.com/user-attachments/assets/449cb1a9-3b2c-47d9-8e8d-ce17ad6f3f23)

*Figure 2: Example of processed answer sheet with detected bubbles and scoring results*

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

MIT License - See [LICENSE](LICENSE) for details.
