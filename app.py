import cv2
import numpy as np
import streamlit as st
from PIL import Image
from PIL import ImageOps
from streamlit_cropper import st_cropper
import io
import google.generativeai as genai
from imutils.perspective import four_point_transform
from helper.perspective_correction import adjust_perspective, adjust_perspective_crop_by_coordinates  # Import the function

# Read Gemini API key from file
try:
    with open('secrets/gemini_api_key.txt', 'r') as f:
        gemini_api_key = f.read().strip()
        genai.configure(api_key=gemini_api_key)
except FileNotFoundError:
    st.error("API key file not found. Please make sure 'secrets/gemini_api_key.txt' exists.")
    st.stop()  # Stop execution if API key is not found

def upload_to_gemini(image_np, mime_type=None):
    """Uploads the given numpy image to Gemini."""
    # Convert numpy array to bytes
    _, image_encoded = cv2.imencode('.jpg', image_np)
    # Create an in-memory bytes buffer
    image_bytes = io.BytesIO(image_encoded) 
    file = genai.upload_file(image_bytes, mime_type=mime_type)
    print(f"Uploaded image as: {file.uri}")
    return file

# --- Answer Key Processing ---
def process_answer_key(answer_key_file):
    """Processes the uploaded answer key file and returns a dictionary of answers."""
    if answer_key_file is not None:
        try:
            # Assuming the answer key is a plain text file with format "1: A, 2: B, ..."
            answer_key_text = answer_key_file.read().decode("utf-8")
            answer_key = {}
            for line in answer_key_text.splitlines():
                q_num, answer = line.split(":")
                answer_key[int(q_num.strip())] = answer.strip()
            return answer_key
        except Exception as e:
            st.error(f"Error processing answer key: {e}")
    return None

# --- Streamlit App ---
st.title("Student Answer Sheet Processor")

# Define custom CSS
custom_css = """
<style>
.stSubheader {
    font-size: 18px;
}
.stFileUploader label {
    font-size: 16px;
}
</style>
"""

# Inject custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)


# --- Answer Key Upload ---
st.subheader("Upload Answer Key")
answer_key_file = st.file_uploader("Choose an answer key file (text file)", type=["txt"])
answer_key = process_answer_key(answer_key_file)

# --- Student Answer Sheet Processing ---
st.subheader("Upload Student Answer Sheet")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # --- Tabs for Displaying and Processing ---
    tab1, tab2, tab3 = st.tabs(["Original Image", "Processed Image", "Gemini Processing & Scoring"])

    with tab1:
        # --- Stage 1: Preprocessing ---
        st.subheader("Stage 1: Preprocessing")

        # Use st.columns to create two columns
        col1, col2 = st.columns(2)

        #with col1:
        # --- Adjust Perspective ---
        img_np = adjust_perspective(img_np)  # Call the function

        # --- Manual Cropping ---
        if "manual_crop_active" not in st.session_state:
            st.session_state.manual_crop_active = False

        if st.button("Crop Image Manually"):
            st.session_state.manual_crop_active = True  # Activate cropping mode

        if st.session_state.manual_crop_active:
            pil_image = Image.fromarray(img_np)

            if "cropped_img" not in st.session_state:
                st.session_state.cropped_img = None

            cropped_img_temp = st_cropper(
                pil_image,
                realtime_update=True,
                box_color="green",
                aspect_ratio=None,
                key="cropper_1",
                return_type="box"  # Change this to "box"
            )

            if st.button("Confirm Crop"):
                if cropped_img_temp is not None:
                    st.session_state.manual_crop_active = False  # Deactivate cropping mode

                    # Get the cropping box coordinates
                    left, top, width, height = cropped_img_temp['left'], cropped_img_temp['top'], cropped_img_temp['width'], cropped_img_temp['height']

                    # Print the box coordinates and angle
                    # print(f"Box Coordinates: Left={left}, Top={top}, Width={width}, Height={height}")

                    # Rotate the original image
                    # Automatically rotate and crop the image
                    rotated_cropped_image = adjust_perspective_crop_by_coordinates(img_np, left, top, width, height)

                    rotated_cropped_image=adjust_perspective(rotated_cropped_image)

                    # Convert to OpenCV format
                    img_np = cv2.cvtColor(np.array(rotated_cropped_image), cv2.COLOR_RGB2BGR)
                    

                else:
                    st.warning("Please select a cropping region first.")

        #with col2: 
        # Display the image after preprocessing
        st.image(img_np, caption="Preprocessed Image", use_container_width=True) 

    with tab2:
        # --- Stage 2: Bubble Detection and Extraction ---
        st.subheader("Stage 2: Bubble Detection and Extraction")

        # Default values for sliders
        min_width = 10
        min_height = 5
        min_aspect_ratio = 0.9
        max_aspect_ratio = 1.1

        # Sliders for bubble detection parameters
        st.subheader("Bubble Detection Parameters")

        # Button to toggle visibility of all sliders
        show_all_sliders = st.checkbox("Fine-tune Parameters", value=False, help="Toggle to show/hide all sliders.")
        # Use st.columns to create a layout with 2 columns
        col1, col2 = st.columns(2)

        with col1:
            if show_all_sliders:
                min_width = st.slider("Minimum Width (pixels)", 1, 20, 10, help="Minimum width of the contour to be considered as a bubble.")
                #min_height = st.slider("Minimum Height (pixels)", 1, 20, 5, help="Minimum height of the contour to be considered as a bubble.")

        with col2:
            if show_all_sliders:
                min_aspect_ratio = st.slider("Minimum Aspect Ratio", 0.5, 1.0, 0.9, help="Minimum aspect ratio (width/height) of the contour to be considered as a bubble. A value closer to 1 means the contour is more circular. For example, 0.9 means the contour can be slightly elongated.")
                #max_aspect_ratio = st.slider("Maximum Aspect Ratio", 1.0, 1.5, 1.1, help="Maximum aspect ratio (width/height) of the contour to be considered as a bubble. A value closer to 1 means the contour is more circular. For example, 1.1 means the contour can be slightly wider than it is tall.")

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

        # Find contours (potential bubbles)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_coords = []  # Store bubble coordinates

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # Filter for bubble-like shapes (adjust these parameters as needed)
            if (not show_all_sliders or (w >= min_width and h >= min_height)) and ar >= min_aspect_ratio and ar <= max_aspect_ratio:
                bubble_coords.append((x, y, w, h))
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Draw for visualization

        # Infer answer area from bubble coordinates
        if bubble_coords:
            # Calculate average bubble size
            avg_width = sum(w for _, _, w, _ in bubble_coords) / len(bubble_coords)
            avg_height = sum(h for _, _, _, h in bubble_coords) / len(bubble_coords)

            # Filter bubbles based on size similarity
            filtered_coords = []
            for x, y, w, h in bubble_coords:
                if abs(w - avg_width) < 0.2 * avg_width and abs(h - avg_height) < 0.2 * avg_height:
                    filtered_coords.append((x, y, w, h))

            if filtered_coords:
                x_min = min(x for x, _, _, _ in filtered_coords)
                y_min = min(y for _, y, _, _ in filtered_coords)
                x_max = max(x + w for x, _, w, _ in filtered_coords)
                y_max = max(y + h for _, y, _, h in filtered_coords)

                # Add some padding to the answer area
                padding = 15
                x_min -= padding + 90
                y_min -= padding
                x_max += padding
                y_max += padding

                # Extract answer area using perspective transform
                answer_area_contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                warped = four_point_transform(img_np.copy(), answer_area_contour.reshape(4, 2))

                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw answer area

                # Divide warped image into 4 columns
                warped_height, warped_width = warped.shape[:2]
                column_width = warped_width // 4
                columns = [
                    warped[0:warped_height, 0:column_width],
                    warped[0:warped_height, column_width:2 * column_width],
                    warped[0:warped_height, 2 * column_width:3 * column_width],
                    warped[0:warped_height, 3 * column_width:warped_width]
                ]

                # Extract header (from top edge to y_min, full width)
                header = img_np[0:y_min, 0:img_np.shape[1]]  # Corrected slicing

            else:
                st.warning("No bubbles with similar size found.")
        else:
            st.warning("No bubbles found.")

        # Display the processed image (after perspective correction and cropping)
        st.image(img_np, caption="Processed Image with Detected Bubbles", use_container_width=True)
        if bubble_coords:
            st.image(warped, caption="Answer Area after Perspective Transform", use_container_width=True)
            
            # Use st.columns to create a layout with 4 columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.image(columns[0], caption="Column 1", use_container_width=True)
            
            with col2:
                st.image(columns[1], caption="Column 2", use_container_width=True)
            
            with col3:
                st.image(columns[2], caption="Column 3", use_container_width=True)
            
            with col4:
                st.image(columns[3], caption="Column 4", use_container_width=True)
            
            st.image(header, caption="Answer Sheet Header", use_container_width=True)
        # ... (display other processed images like warped, columns, header) ...

    with tab3:
        # --- Gemini API Integration and Scoring ---
        st.subheader("Gemini Processing and Scoring")
        model_name = st.selectbox("Select Gemini Model", ["gemini-1.5-pro-latest", "gemini-1.5-flash"])

        if st.button("Process with Gemini"):
            if answer_key is None:
                st.error("Please upload an answer key first.")
            elif not bubble_coords:
                st.error("No bubbles were detected. Please check the image and try again.")
            else:
                all_extracted_answers = []
                for i, answer_column in enumerate(columns):  # Use the extracted columns
                    file = upload_to_gemini(answer_column, mime_type="image/jpeg")

                    # Create the model
                    generation_config = {
                        "temperature": 0,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "response_mime_type": "text/plain",
                    }

                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config,
                    )

                    # Construct the prompt with the image
                    prompt = [
                        file,
                        "Extract the selected answers from the provided answer sheet with question numbers. "
                        "Detect any marks in the bubbles (fully filled, partially filled, or lightly shaded), "
                        "associate them with their respective question numbers, and determine the selected answer option (A, B, C, or D). "
                        "Remember to look closely for each question before responding. "
                        "Present the results in the format:\n1: A,\n2: B,\n3: C, ...\n"
                    ]

                    # Generate the response
                    response = model.generate_content(prompt)

                    # Display response
                    st.write(f"Response for Answer Section {i+1}:")
                    st.write(response.text)
                    
                    # --- Extract answers from Gemini response ---
                    extracted_answers = {} 
                    try:
                        for line in response.text.splitlines():
                            q_num, answer = line.split(":")
                            extracted_answers[int(q_num.strip())] = answer.strip()
                        all_extracted_answers.extend(list(extracted_answers.values())) 
                    except Exception as e:
                        st.error(f"Error extracting answers from Gemini response: {e}")
                        st.write("Please make sure the response is in the correct format (e.g., '1: A, 2: B, ...')")
                        continue

                # --- Calculate Score ---
                if len(all_extracted_answers) == len(answer_key):
                    correct_answers = sum(a == b for a, b in zip(all_extracted_answers, answer_key.values()))
                    score = (correct_answers / len(answer_key)) * 100
                    st.success(f"Score: {score:.2f}%")

                    st.write("Correct Answers:", correct_answers)
                else:
                    st.error("Number of extracted answers does not match the answer key.")