# AutoMQA: Automated Multiple-Choice Assessment

AutoMQA is a tool that leverages Large Language Model (LLM) technology to automate the grading of multiple-choice assessments. It takes an image of a student's answer sheet as input and outputs the student's score.

## Features

* **Automated Grading:**  Eliminates the need for manual grading of multiple-choice tests.
* **Image Recognition:**  Utilizes Optical Character Recognition (OCR) to process answer sheets from uploaded images.
* **LLM-Powered Analysis:** Employs LLMs to interpret student responses and match them against the correct answers, even with slight variations or misspellings.
* **Efficient and Scalable:**  Can handle a large volume of answer sheets, saving time and effort for educators.
* **Flexible Answer Key:** Supports various answer key formats (e.g., plain text, CSV) for easy integration.
* **Detailed Reporting:** Provides a breakdown of student performance, including individual question scores and overall accuracy.

## How it Works

1. **Image Upload:**  Users upload an image of the completed answer sheet.
2. **OCR Processing:** The system uses OCR to extract the student's responses from the image.
3. **Response Preprocessing:**  LLMs analyze the extracted text, identify answer choices, and standardize formatting for accurate matching.
4. **Answer Key Matching:** The processed responses are compared against a provided answer key.
5. **Result Generation:**  The system calculates the student's score and provides a detailed report.

## Technology Stack

* **Python:**  Core programming language for the application.
* **Optical Character Recognition (OCR):**  Libraries like Tesseract or Google Cloud Vision API are used for text extraction from images.
* **Large Language Models (LLMs):**  LLMs like GPT-4 are used for understanding and interpreting student responses.
* **(Optional) Web Framework:**  Flask or Django can be used to build a web interface for the application.

## Getting Started

1. **Clone the repository:** `git clone https://github.com/your-username/AutoMQA.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Set up your answer key:** Create a file (e.g., `answer_key.txt`) containing the correct answers for your assessment. For example: 1. A; 2. B; 3. C
4. **Run the application:** 
```bash
python main.py --image path/to/answer_sheet.jpg --key answer_key.txt
```
## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)   
 file for details.
