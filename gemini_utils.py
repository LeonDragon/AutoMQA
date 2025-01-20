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

def recheck_single_column(column_array, model_name, answer_key_path):
    """Recheck a single column with different parameters and prompt"""
    try:
        print(f"\n=== Starting Recheck API Call ===")
        print(f"Model: {model_name}")
        print(f"Temperature: 0.7")
        print(f"Column shape: {column_array.shape}")
        
        # Create a new Gemini model instance for rechecking
        genai.configure(api_key=gemini_api_key)
        
        # Different generation config for rechecking
        generation_config = {
            "temperature": 0.7,  # More creative interpretations
            "top_p": 0.9,        # Wider range of possibilities
            "top_k": 40,         # Consider more options
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Create model instance
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
        
        # Upload image
        file = upload_to_gemini(column_array, mime_type="image/jpeg")

        # Different prompt for rechecking
        prompt = [
            file,
            "You are rechecking an answer sheet column. Please carefully analyze the image again and: \n"
            "1. Look for any marks you may have missed previously\n"
            "2. Consider alternative interpretations of ambiguous marks\n" 
            "3. Pay special attention to faint or partial marks\n"
            "4. If you're unsure between two options, choose the more likely one\n"
            "5. Return results in JSON format with question numbers as keys and answers as values:\n"
            "{\n"
            "  \"1\": \"A\",  // Clearly marked A\n"
            "  \"2\": \"B\",  // Clearly marked B\n"  
            "  \"3\": \"X\",  // No mark or too faint\n"
            "  ...\n"
            "}\n"
            "6. Be more lenient in your interpretation this time\n"
            "7. If you see any marks that could reasonably be interpreted as an answer, include them\n"
        ]

        # Get response with usage tracking
        response = model.generate_content(prompt)
        json_response = json.loads(response.text)
        
        # Get token usage from response
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        # Load answer key
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)["answerKeys"]

        # Calculate scores
        scores = {}
        for test_code, test_answer_key in answer_key_data.items():
            correct_answers = sum(1 for q_num, answer in json_response.items() 
                               if str(q_num) in test_answer_key and answer == test_answer_key[str(q_num)])
            score = (correct_answers / len(test_answer_key)) * 100
            scores[test_code] = score

        return {
            'answers': json_response,
            'scores': scores,
            'response': json_response,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens
            }
        }
    except Exception as e:
        return {'error': str(e)}

def process_single_column(column_array, model_name, answer_key_path, temperature=0):
    """Process a single column with Gemini"""
    try:
        # Create a new Gemini model instance for each thread
        genai.configure(api_key=gemini_api_key)
        
        # Create generation config with adjustable temperature
        generation_config = {
            "temperature": temperature,  # Allow some randomness for rechecking
            "top_p": 0.95,      # Slightly less strict sampling
            "top_k": 40,        # Wider range of options
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Create model instance
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
        
        # Initialize token counters
        input_tokens = 0
        output_tokens = 0

        # Upload image
        file = upload_to_gemini(column_array, mime_type="image/jpeg")

        # Create prompt
        prompt = [
            file,
            "You are an expert at reading and interpreting multiple choice answer sheets. "
            "Carefully analyze the provided image of an answer sheet column and extract the selected answers. "
            "Follow these guidelines:\n"
            "1. Each question has 4 bubbles labeled A, B, C, D from left to right\n"
            "2. Look for any marks in the bubbles - they could be fully filled, partially filled, or lightly shaded\n"
            "3. If multiple bubbles are marked for a question, choose the darkest/most filled one\n"
            "4. If no bubbles are marked or the marks are too faint, return 'X' for that question\n"
            "5. Return results in JSON format with question numbers as keys and answers as values:\n"
            "{\n"
            "  \"1\": \"A\",  // Clearly marked A\n"
            "  \"2\": \"B\",  // Clearly marked B\n"  
            "  \"3\": \"X\",  // No mark or too faint\n"
            "  ...\n"
            "}\n"
            "6. Be extremely careful with question numbers - double check they are correct\n"
            "7. If you're unsure about any answer, mark it as 'X' rather than guessing\n"
        ]

        # Get response with usage tracking
        response = model.generate_content(prompt)
        json_response = json.loads(response.text)
        
        # Get token usage from response
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        # Load answer key
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)["answerKeys"]

        # Calculate scores
        scores = {}
        for test_code, test_answer_key in answer_key_data.items():
            correct_answers = sum(1 for q_num, answer in json_response.items() 
                               if str(q_num) in test_answer_key and answer == test_answer_key[str(q_num)])
            score = (correct_answers / len(test_answer_key)) * 100
            scores[test_code] = score

        return {
            'answers': json_response,
            'scores': scores,
            'response': json_response,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens
            }
        }
    except Exception as e:
        return {'error': str(e)}

def process_student_answers(columns, model_name, answer_key_path):
    """Process columns in parallel using ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Load answer key data first
    try:
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)["answerKeys"]
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return None, None, None, None
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_column, col, model_name, answer_key_path): idx
            for idx, col in enumerate(columns)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                results.append((idx, {'error': str(e)}))
    
    # Sort results by original column order
    results.sort(key=lambda x: x[0])
    
    # Combine results
    combined_answers = {}
    combined_scores = {}
    all_responses = []
    has_error = False
    
    for idx, result in results:
        if 'error' in result:
            has_error = True
            break
        combined_answers.update(result['answers'])
        combined_scores.update(result['scores'])
        all_responses.append(result['response'])
    
    if has_error:
        return None, None, None, None, None
        
    # Calculate total token usage
    total_input_tokens = sum(result.get('tokens', {}).get('input', 0) for _, result in results)
    total_output_tokens = sum(result.get('tokens', {}).get('output', 0) for _, result in results)
    
    return answer_key_data, list(combined_answers.values()), combined_scores, all_responses, {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }
