<div align="center">
  <img src="static/images/UEHU_huyhieu.png" alt="AutoMQA Logo" width="150"/>

  # AutoMQA: Automated Multiple-Choice Assessment

  [![GitHub release](https://img.shields.io/github/release/your-username/AutoMQA.svg?style=flat-square)](https://github.com/your-username/AutoMQA/releases)
  [![License](https://img.shields.io/github/license/your-username/AutoMQA.svg?style=flat-square)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/your-username/AutoMQA.svg?style=flat-square)](https://github.com/your-username/AutoMQA/stargazers)
  [![GitHub issues](https://img.shields.io/github/issues/your-username/AutoMQA.svg?style=flat-square)](https://github.com/your-username/AutoMQA/issues)

</div>

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

AutoMQA is a Flask-based web application that automates the grading of multiple-choice assessments using computer vision and AI. It processes scanned answer sheets, detects and analyzes answer bubbles, and evaluates responses using Gemini AI.

## ✨ Features

* **Automated Grading:** Eliminates manual grading of multiple-choice tests
* **Image Processing:** Uses OpenCV for perspective correction, bubble detection, and image enhancement
* **AI-Powered Evaluation:** Leverages Gemini AI for accurate answer evaluation
* **Web Interface:** Provides REST API endpoints for easy integration
* **File Format Support:** Handles JPG, PNG, and HEIC formats
* **Detailed Reporting:** Provides processed images, detected answers, and scoring breakdown

![Application Interface](https://github.com/user-attachments/assets/37699322-d593-4584-8b7c-ca767fdef17c)

*Figure 1: Main application interface showing image upload and processing controls*

![Processing Results](https://github.com/user-attachments/assets/449cb1a9-3b2c-47d9-8e8d-ce17ad6f3f23)

*Figure 2: Example of processed answer sheet with detected bubbles and scoring results*

## 🛠️ Technology Stack

* **Python 3**
* **Flask** - Web framework
* **OpenCV** - Image processing and computer vision
* **Gemini AI** - Answer evaluation and scoring
* **NumPy** - Numerical computations
* **Pillow** - Image manipulation

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/LeonDragon/AutoMQA.git
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

## 📚 API Documentation

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

## 📝 File Requirements

* Supported formats: JPG, PNG, HEIC
* Maximum file size: 16MB
* Answer sheets should have:
  - Clear bubble markings
  - Minimal skew or distortion
  - Good contrast between bubbles and background

## 💻 Usage

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

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Reporting Issues
- Check existing issues before creating new ones
- Provide detailed description of the problem
- Include steps to reproduce
- Attach relevant screenshots or logs

### Feature Requests
- Explain the proposed feature
- Describe potential use cases
- Suggest implementation approach if possible

### Pull Requests
1. Fork the repository
2. Create a new branch for your feature/bugfix
3. Write clear commit messages
4. Add/update tests if applicable
5. Update documentation
6. Submit a pull request with detailed description

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for public methods
- Keep functions small and focused

Contributions are welcome! Please open issues or submit pull requests.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
