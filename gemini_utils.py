# gemini_utils.py
import cv2
import io
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content  # Import for JSON schema
import json  # Import the json library
import time

# Read Gemini API key from file (you might need to adjust the path)
try:
    with open('secrets/gemini_api_key.txt', 'r') as f:
      api_keys = [line.strip() for line in f if line.strip()] 

    if not api_keys:
      raise ValueError("No API keys found in gemini_api_key.txt")
    
except FileNotFoundError:
    print("API key file not found. Please make sure 'secrets/gemini_api_key.txt' exists.")
    exit()  # Exit if API key is not found

def compress_image(image_np, quality=50):
    """Compress image using OpenCV with specified quality percentage"""
    # Step 1: Convert the original image_np to a JPEG version (uncompressed)
    _, original_jpg = cv2.imencode('.jpeg', image_np)

    # Step 2: Compress the image with the specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, image_encoded = cv2.imencode('.jpeg', image_np, encode_param)

    # Step 3: Calculate compression ratio
    original_size = len(original_jpg)  # Size of the uncompressed JPEG
    compressed_size = len(image_encoded)  # Size of the compressed JPEG
    ratio = compressed_size / original_size * 100

    # Step 4: Print results
    print(f"Compressed image: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB ({ratio:.1f}%)")
    return image_encoded

def upload_to_gemini(image_np, mime_type=None, compress_quality=50):
    """Uploads the given numpy image to Gemini with optional compression"""
    if compress_quality < 100:
        image_encoded = compress_image(image_np, compress_quality)
    else:
        _, image_encoded = cv2.imencode('.jpg', image_np)
        
    image_bytes = io.BytesIO(image_encoded) 
    file = genai.upload_file(image_bytes, mime_type=mime_type)
    print(f"Uploaded image as: {file.uri}")
    return file

def process_answer_key(answer_key_image):
    """Processes the uploaded answer key image using Gemini and returns a dictionary of answers."""
    if answer_key_image is not None:
        try:
            file = upload_to_gemini(answer_key_image, mime_type="image/jpeg", compress_quality=50)

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

            from prompts import get_prompt
            
            # Construct the prompt with the image
            prompt = [
                file,
                get_prompt('default', 'answer_key')
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
        
        # Upload image with compression
        file = upload_to_gemini(column_array, mime_type="image/jpeg", compress_quality=50)

        from prompts import get_prompt
            
        # Different prompt for rechecking
        prompt = [
            file,
            get_prompt('default', 'recheck_analysis')
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

def call_gemini_api(prompt_content, model_name="gemini-1.5-flash-latest", 
                   temperature=0, mime_type="application/json", 
                   image_np=None, compress_quality=100):
    """Make API call to Gemini with optional image upload"""
    try:
        # Configure API
        import random
        random_key = random.choice(api_keys)
        genai.configure(api_key=random_key)
        print(f"Generative AI configured with a random API key ending in: {random_key[-5:]}")
        genai.configure(api_key=random_key)
        
        # Create generation config
        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": mime_type,
        }

        # Create model instance
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

        # Build prompt content
        prompt = []
        if image_np is not None:
            file = upload_to_gemini(image_np, mime_type="image/jpeg", compress_quality=compress_quality)
            prompt.append(file)
        prompt.append(prompt_content)

        # Get response with usage tracking
        response = model.generate_content(prompt)
        
        # Get token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        return {
            'response': response,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens
            },
            'rand_api_key': random_key[-5:]
        }
    except Exception as e:
        return {'error': str(e)}

def process_single_column_OLD(column_array, model_name, answer_key_path, 
                         temperature=0, 
                         compress_quality=100):
    """Process a single column with Gemini"""
    try:
        # Get prompt content
        from prompts import get_prompt
        prompt_content = get_prompt('experiment_3', 'column_analysis')

        # First API call - column analysis
        print("\n== Reponse of Prompt 1 ==")
        api_result = call_gemini_api(
            prompt_content=prompt_content,
            model_name=model_name,
            temperature=0,
            image_np=column_array,
            compress_quality=compress_quality,
            mime_type='text/plain'
            #mime_type="application/json"
        )
        print(api_result['response'].text)
        
        if 'error' in api_result:
            return {'error': api_result['error']}

        # Second API call - JSON extraction and validation
        from prompts import get_prompt
        json_prompt = get_prompt('json_extract', 'json')
        
        # Chain the first response into the second prompt
        print("\n== Reponse of Prompt 2 ==")
        json_result = call_gemini_api(
            prompt_content=f"{api_result['response'].text}\n\n{json_prompt}",
            #model_name='gemini-1.5-flash-8b',
            model_name=model_name,
            temperature=0,  # Use 0 temperature for strict JSON extraction
            mime_type="application/json"
        )
        print(json_result['response'].text)
                
        
        # If use first prompt only
        #json_result = api_result

        if 'error' in json_result:
            return {'error': json_result['error']}

        # Parse the final JSON response
        json_response = json.loads(json_result['response'].text)
        
        # Combine token counts from both API calls
        total_tokens = {
            'input': api_result['tokens']['input'] + json_result['tokens']['input'],
            'output': api_result['tokens']['output'] + json_result['tokens']['output']
        }
        
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
            'tokens': total_tokens
        }
    except Exception as e:
        print("===>> Error in process_single_column")
        return {'error': str(e)}
    
def process_single_column(column_array, model_name, answer_key_path,
                          temperature=0,
                          compress_quality=100, max_retries=5, retry_delay=5):
    """Process a single column with Gemini, with retries for API calls."""
    retries_api1 = 0
    retries_api2 = 0

    while retries_api1 < max_retries:
        try:
            # Get prompt content
            from prompts import get_prompt
            prompt_content = get_prompt('experiment_4', 'column_analysis')

            # First API call - column analysis
            print("\n== Response of Prompt 1 ==")
            api_result = call_gemini_api(
                prompt_content=prompt_content,
                model_name=model_name,
                temperature=0,
                image_np=column_array,
                compress_quality=compress_quality,
                mime_type='text/plain'
                # mime_type="application/json"
            )
            print(api_result['response'].text)

            if 'error' in api_result:
                raise Exception(api_result['error']) # Raise an exception to trigger the retry mechanism

            break  # API call successful, break out of the retry loop

        except Exception as e:
            retries_api1 += 1
            print(f"Error in first API call: {e}. Retrying in {retry_delay} seconds... (Attempt {retries_api1}/{max_retries})")
            if retries_api1 < max_retries:
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for first API call. Returning error.")
                return {'error': str(e)}

    while retries_api2 < max_retries:
        try:
            # Second API call - JSON extraction and validation
            from prompts import get_prompt
            json_prompt = get_prompt('json_extract', 'json')

            # Chain the first response into the second prompt
            print("\n== Response of Prompt 2 ==")
            json_result = call_gemini_api(
                prompt_content=f"{api_result['response'].text}\n\n{json_prompt}",
                model_name='gemini-1.5-flash-8b',
                #model_name=model_name,
                temperature=0,  # Use 0 temperature for strict JSON extraction
                mime_type="application/json"
            )
            print(json_result['response'].text)

            if 'error' in json_result:
              raise Exception(json_result['error'])

            break

        except Exception as e:
            retries_api2 += 1
            print(f"Error in second API call: {e}. Retrying in {retry_delay} seconds... (Attempt {retries_api2}/{max_retries})")
            if retries_api2 < max_retries:
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for second API call. Returning error.")
                return {'error': str(e)}

    try:
        # Parse the final JSON response
        json_response = json.loads(json_result['response'].text)

        # Combine token counts from both API calls
        total_tokens = {
            'input': api_result['tokens']['input'] + json_result['tokens']['input'],
            'output': api_result['tokens']['output'] + json_result['tokens']['output']
        }

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
            'tokens': total_tokens
        }
    except Exception as e:
        print("===>> Error in process_single_column (after API calls)")
        return {'error': str(e)}
    
def process_single_column_CoT_01(column_array, model_name, answer_key_path, 
                         temperature=0, 
                         compress_quality=100):
    """Process a single column with Gemini"""
    try:
        # Get prompt content
        from prompts import get_prompt
        prompt_content = get_prompt('experiment_3', 'column_analysis')

        # First API call - column analysis
        print("\n== Reponse of Prompt 1 ==")
        api_result = call_gemini_api(
            prompt_content=prompt_content,
            model_name=model_name,
            temperature=0,
            image_np=column_array,
            compress_quality=compress_quality,
            mime_type='text/plain'
            #mime_type="application/json"
        )
        print(api_result['response'].text)
        
        if 'error' in api_result:
            return {'error': api_result['error']}

        # # Second API call - JSON extraction and validation
        # from prompts import get_prompt
        # json_prompt = get_prompt('json_extract', 'json')
        
        # # Chain the first response into the second prompt
        # print("\n== Reponse of Prompt 2 ==")
        # json_result = call_gemini_api(
        #     prompt_content=f"{api_result['response'].text}\n\n{json_prompt}",
        #     model_name=model_name,
        #     temperature=0,  # Use 0 temperature for strict JSON extraction
        #     mime_type="application/json"
        # )
        # print(json_result['response'].text)
                
        
        # If use first prompt only
        json_result = api_result

        if 'error' in json_result:
            return {'error': json_result['error']}

        # Parse the final JSON response
        json_response = json.loads(json_result['response'].text)
        
        # Combine token counts from both API calls
        total_tokens = {
            'input': json_result['tokens']['input'],
            'output': json_result['tokens']['output']
        }
        
        # Load answer key
        # with open(answer_key_path, 'r') as f:
        #     answer_key_data = json.load(f)["answerKeys"]

        # Calculate scores
        scores = {}
        # for test_code, test_answer_key in answer_key_data.items():
        #     correct_answers = sum(1 for q_num, answer in json_response.items() 
        #                        if str(q_num) in test_answer_key and answer == test_answer_key[str(q_num)])
        #     score = (correct_answers / len(test_answer_key)) * 100
        #     scores[test_code] = score

        return {
            'answers': json_response,
            'scores': scores,
            'response': json_response,
            'tokens': total_tokens,
            'llmReponse_CoT': api_result
        }
    except Exception as e:
        return {'error': str(e)}

def process_single_column_chaining_02(llmReponse_CoT, tokens, model_name, answer_key_path, 
                         temperature=0, 
                         compress_quality=100):
    """Process a prompt chaining with Gemini"""
    try:
        # llmReponse_CoT => String of prompt with append all previous responses from LLM

        # Second API call - JSON extraction and validation
        from prompts import get_prompt
        json_prompt = get_prompt('json_extract', 'json')
        
        # Chain the first response into the second prompt
        print("\n== Reponse of Prompt 2 ==")
        json_result = call_gemini_api(
            prompt_content=f"{llmReponse_CoT}\n\n{json_prompt}",
            model_name=model_name,
            temperature=0,  # Use 0 temperature for strict JSON extraction
            mime_type="application/json"
        )
        print(json_result['response'].text)

        if 'error' in json_result:
            return {'error': json_result['error']}

        # Parse the final JSON response
        json_response = json.loads(json_result['response'].text)
        
        # Combine token counts from both API calls
        total_tokens = {
            'input': tokens.input + json_result['tokens']['input'],
            'output': tokens.output + json_result['tokens']['output']
        }
        
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
            'tokens': total_tokens
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
    with ThreadPoolExecutor(max_workers=12) as executor:
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
    
    print("================== RESULTS ==================")
    print(results)

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

# NOT NEED
def process_single_vertical_group(group, model_name, answer_key_path, 
                          temperature=0, 
                          compress_quality=100):
    """Process a single vertical group (portion of a column) with Gemini."""
    try:
        # 1. Extract image data from the group
        image_base64 = group['image']
        image_data = base64.b64decode(image_base64)
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # 2. Get the group number and column number
        group_num = group['group']
        column_num = group['column']

        # 3. Get prompt content for the first API call (column analysis)
        from prompts import get_prompt
        prompt_content = get_prompt('experiment_3', 'column_analysis')

        # 4. First API call - column analysis on the vertical group
        print(f"\n== Response of Prompt 1 (Column Analysis - Column {column_num}, Group {group_num}) ==")
        api_result = call_gemini_api(
            prompt_content=prompt_content,
            model_name=model_name,
            temperature=temperature,  # Use the provided temperature
            image_np=image_array,
            compress_quality=compress_quality,
            mime_type='text/plain'
        )
        print(api_result['response'].text)

        if 'error' in api_result:
            return {'error': api_result['error']}

        # 5. Get prompt content for the second API call (JSON extraction)
        json_prompt = get_prompt('json_extract', 'json')

        # 6. Second API call - JSON extraction and validation
        print(f"\n== Response of Prompt 2 (JSON Extraction - Column {column_num}, Group {group_num}) ==")
        json_result = call_gemini_api(
            prompt_content=f"{api_result['response'].text}\n\n{json_prompt}",
            model_name=model_name,
            temperature=0,  # Use 0 temperature for strict JSON extraction
            mime_type="application/json"
        )
        print(json_result['response'].text)

        if 'error' in json_result:
            return {'error': json_result['error']}

        # 7. Parse the final JSON response
        json_response = json.loads(json_result['response'].text)

        # 8. Combine token counts from both API calls
        total_tokens = {
            'input': api_result['tokens']['input'] + json_result['tokens']['input'],
            'output': api_result['tokens']['output'] + json_result['tokens']['output']
        }

        # 9. Load answer key (only if needed for scoring in this function)
        # with open(answer_key_path, 'r') as f:
        #     answer_key_data = json.load(f)["answerKeys"]

        # 10. Calculate scores (or move this to the main function if you prefer)
        # scores = {}
        # for test_code, test_answer_key in answer_key_data.items():
        #     correct_answers = sum(1 for q_num, answer in json_response.items()
        #                           if str(q_num) in test_answer_key and answer == test_answer_key[str(q_num)])
        #     score = (correct_answers / len(test_answer_key)) * 100
        #     scores[test_code] = score

        # 11. Return the results, including column and group numbers
        return {
            'answers': json_response,
            #'scores': scores,  # Include if you are calculating scores here
            'response': json_response, # Consider if you need raw or processed response
            'tokens': total_tokens,
            'column': column_num, # Add column number to the result for identification later
            'group': group_num # Add group number if needed
        }
    except Exception as e:
        print(f"Error processing group: {e}")
        return {'error': str(e)}

# NOT NEED  
# New approach for split each column further
def process_student_answers_vertical_group(vertical_groups, model_name, answer_key_path):
    """Process vertical groups in parallel using ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load answer key data first
    try:
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)["answerKeys"]
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return None, None, None, None, None  # Return None for all outputs on error

    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(process_single_vertical_group, group, model_name, answer_key_path): group.get('column') # Pass column number for sorting
            for group in vertical_groups
        }

        for future in as_completed(futures):
            col = futures[future] # Get the column number
            try:
                result = future.result()
                results.append((col, result))  # Store column number with result
            except Exception as e:
                results.append((col, {'error': str(e)}))

    # Sort results by original column order
    results.sort(key=lambda x: x[0])

    # Combine results from the same column
    combined_results = {}
    for col, result in results:
        if 'error' in result:
            return None, None, None, None, None # Return None for all outputs on error

        if col not in combined_results:
            combined_results[col] = {'answers': {}, 'scores': {}, 'response': [], 'tokens': {'input': 0, 'output': 0}}

        combined_results[col]['answers'].update(result['answers'])
        combined_results[col]['scores'].update(result['scores'])
        combined_results[col]['response'].append(result['response'])
        combined_results[col]['tokens']['input'] += result.get('tokens', {}).get('input', 0)
        combined_results[col]['tokens']['output'] += result.get('tokens', {}).get('output', 0)

    # Aggregate results from all columns
    combined_answers = {}
    combined_scores = {}
    all_responses = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Sort columns to maintain order
    sorted_columns = sorted(combined_results.keys())

    for col in sorted_columns:
        combined_answers.update(combined_results[col]['answers'])
        combined_scores.update(combined_results[col]['scores'])
        all_responses.extend(combined_results[col]['response'])
        total_input_tokens += combined_results[col]['tokens']['input']
        total_output_tokens += combined_results[col]['tokens']['output']

    return answer_key_data, list(combined_answers.values()), combined_scores, all_responses, {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }