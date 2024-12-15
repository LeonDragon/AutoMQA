# gemini_utils.py
import cv2
import io
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content  # Import for JSON schema
import json  # Import the json library

# Read Gemini API key from file (you might need to adjust the path)
try:
    with open('secrets/gemini_api_key.txt', 'r') as f:
        gemini_api_key = f.read().strip()
        genai.configure(api_key=gemini_api_key)
except FileNotFoundError:
    print("API key file not found. Please make sure 'secrets/gemini_api_key.txt' exists.")
    exit()  # Exit if API key is not found

def upload_to_gemini(image_np, mime_type=None):
    """Uploads the given numpy image to Gemini."""
    _, image_encoded = cv2.imencode('.jpg', image_np)
    image_bytes = io.BytesIO(image_encoded) 
    file = genai.upload_file(image_bytes, mime_type=mime_type)
    print(f"Uploaded image as: {file.uri}")
    return file

def process_answer_key(answer_key_image):
    """Processes the uploaded answer key image using Gemini and returns a dictionary of answers."""
    if answer_key_image is not None:
        try:
            file = upload_to_gemini(answer_key_image, mime_type="image/jpeg")

            # Create the model
            generation_config = {
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_schema": content.Schema(
                    type = content.Type.OBJECT,
                    enum = [],
                    required = ["exam", "answer_keys"],
                    properties = {
                        "exam": content.Schema(
                            type = content.Type.OBJECT,
                            enum = [],
                            required = ["title", "code"],
                            properties = {
                            "title": content.Schema(
                                type = content.Type.STRING,
                                description = "Title of the exam",
                            ),
                            "code": content.Schema(
                                type = content.Type.STRING,
                                description = "Exam code",
                                ),
                            },
                        ),
                        "answer_keys": content.Schema(
                            type = content.Type.ARRAY,
                            items = content.Schema(
                                type = content.Type.OBJECT,
                                enum = [],
                                required = ["test_code", "answers"],
                                properties = {
                                    "test_code": content.Schema(
                                        type = content.Type.STRING,
                                        description = "Code of the answer key (e.g., HTT105327)",
                                        ),
                                    "answers": content.Schema(
                                        type = content.Type.ARRAY,
                                        items = content.Schema(
                                            type = content.Type.OBJECT,
                                            enum = [],
                                            required = ["question_number", "answer"],
                                            properties = {
                                                "question_number": content.Schema(
                                                    type = content.Type.INTEGER,
                                                    description = "Question number",
                                                ),
                                                "answer": content.Schema(
                                                    type = content.Type.STRING,
                                                    description = "Answer choice (A, B, C, or D)",
                                                ),
                                            },
                                        ),
                                    ),
                                },
                            ),
                        ),
                    },
                ),
                "response_mime_type": "application/json",
                }

            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config=generation_config,
            )

            # Construct the prompt with the image
            prompt = [
                file,
                "Extract the answer key from the provided image. There are total 60 questions for each test code "
            ]

            # Generate the response
            response = model.generate_content(prompt)

            # (Optional) Display response in the app.py
            # st.write(f"Response for Answer Key:")
            #print(response.text)

            # Parse the JSON response
            answer_key_data = json.loads(response.text)["answer_keys"]
            print("====== Aanswer_key_list ======")
            print(answer_key_data)
            
            answer_keys = {}  # Initialize a dictionary to store answer keys for each test_code
            for answer_key_set in answer_key_data:
                test_code = answer_key_set["test_code"]
                answer_key = {item["question_number"]: item["answer"] for item in answer_key_set["answers"]}
                answer_keys[test_code] = answer_key  # Store answer key with corresponding test_code


            return answer_key

        except Exception as e:
            # (Optional) Display error in the app.py
            # st.error(f"Error processing answer key: {e}")
            print(f"Error processing answer key: {e}")  # Print error in gemini_utils.py
    return None

def process_student_answers(columns, model_name, answer_key_path):
    """Processes the student answer sheet columns using the specified Gemini model and returns the extracted answers and score."""
    print("\n=== Starting Student Answer Processing ===")
    print(f"Model: {model_name}")
    print(f"Number of columns: {len(columns)}")
    print(f"Answer key path: {answer_key_path}")

    all_responses = []

    try:
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)["answerKeys"]
            print(f"Loaded answer key data: {answer_key_data}")
    except FileNotFoundError:
        print(f"Answer key file not found: {answer_key_path}")
        return None, None
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return None, None

    if not columns:
        print("No answer columns were detected. Please check the image and try again.")
        return None, None
    else:
        all_extracted_answers = []
        for i, answer_column in enumerate(columns):
            print(f"\nProcessing Column {i+1}:")
            print(f"Column shape: {answer_column.shape}")
            
            try:
                file = upload_to_gemini(answer_column, mime_type="image/jpeg")
                print("Successfully uploaded column to Gemini")

                generation_config = {
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                }

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                )
                print("\n+ Created Gemini model instance")

                prompt = [
                    file,
                    "Extract the selected answers from the provided answer sheet with question numbers. "
                    "Detect any marks in the bubbles (fully filled, partially filled, or lightly shaded), "
                    "associate them with their respective question numbers, and determine the selected answer option (A, B, C, or D). "
                    "Remember to look closely for each question before responding. "
                    "Present the results in JSON format as follows:\n"
                    "{\n"
                    "  \"1\": \"A\",\n"
                    "  \"2\": \"B\",\n"
                    "  \"3\": \"C\",\n"
                    "  ...\n"
                    "}\n"
                    "Do not include any introductory or concluding remarks, explanations, interpretations, or summaries. Only provide the answer key in the specified JSON format."
                ]

                print("+ Sending request to Gemini...")
                response = model.generate_content(prompt)
                print("\nGemini Response:")
                print("----------------")
                print(response.text)
                print("----------------")

                json_response = json.loads(response.text)
                all_responses.append(json_response)

                extracted_answers = {} 
                # try:
                #     for line in response.text.splitlines():
                #         q_num, answer = line.split(":")
                #         extracted_answers[int(q_num.strip())] = answer.strip()
                #     print(f"Extracted answers: {extracted_answers}")
                #     all_extracted_answers.extend(list(extracted_answers.values())) 
                # except Exception as e:
                #     print(f"Error extracting answers from Gemini response: {e}")
                #     print("Please make sure the response is in the correct format (e.g., '1: A, 2: B, ...')")
                #     return None, None
            except Exception as e:
                print(f"Error processing column {i+1}: {e}")
                return None, None

        print("\nAll columns processed")
        print(f"Total extracted answers: {len(all_extracted_answers)}")
        print(f"Answers: {all_extracted_answers}")

        scores = {}
        for test_code, test_answer_key in answer_key_data.items():
            if len(all_extracted_answers) == len(test_answer_key):
                correct_answers = sum(a == b for a, b in zip(all_extracted_answers, test_answer_key.values()))
                score = (correct_answers / len(test_answer_key)) * 100
                scores[test_code] = score
                print(f"\nScores for test {test_code}:")
                print(f"Correct answers: {correct_answers}/{len(test_answer_key)}")
                print(f"Score: {score}%")
            else:
                print(f"\nNumber of extracted answers does not match the answer key for test_code: {test_code}")
                print(f"Expected {len(test_answer_key)} answers, got {len(all_extracted_answers)}")
                scores[test_code] = None

        return answer_key_data, all_extracted_answers, scores, all_responses