# First, align the image correctly (rotate image)
# Second, try to split each 4 images
# Third, try to cutoff edges to get center of the answers.

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

def detect_sections(image):
    # Convert to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Adjust these parameters if needed ---
    min_header_area = 5000
    min_answer_area = 10000
    header_aspect_ratio_range = (2, 6)  # (min, max)
    answer_aspect_ratio_range = (1.5, 6)  # (min, max)

    header_box = None
    answer_box = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        if (header_aspect_ratio_range[0] < aspect_ratio < header_aspect_ratio_range[1] and 
                area > min_header_area and 
                y < open_cv_image.shape[0] // 3):  # Header criteria
            header_box = (x, y, w, h)
        elif (answer_aspect_ratio_range[0] < aspect_ratio < answer_aspect_ratio_range[1] and 
              area > min_answer_area and 
              y > open_cv_image.shape[0] // 3):  # Answer section criteria
            answer_box = (x, y, w, h)

        if header_box and answer_box:
            break

    # Extract sections
    header = open_cv_image[header_box[1]:header_box[1]+header_box[3], header_box[0]:header_box[0]+header_box[2]] if header_box else None
    answers = open_cv_image[answer_box[1]:answer_box[1]+answer_box[3], answer_box[0]:answer_box[0]+answer_box[2]] if answer_box else None

    return header, answers, header_box, answer_box, thresh


# Streamlit App
st.title("Student Answer Sheet Processor")
st.write("Upload an image of the answer sheet.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    img_np = np.array(image)  # Convert to NumPy array for OpenCV
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process the image to get initial bounding boxes
    st.write("Processing the image...")
    header, answers, header_box, answer_box, thresh = detect_sections(image)

    # --- Interactive Section Selection with Cropper --- 
    st.subheader("Select Sections")

    # --- Header Selection --- 
    st.write("Select the Header Section")
    cropped_header = st_cropper(
        image,
        realtime_update=True,
        box_color='#00FF00',  # Green color for header box
        aspect_ratio=None,  # No aspect ratio constraint
    )
    # Convert cropped image to OpenCV format
    header_np = np.array(cropped_header)
    header_np = cv2.cvtColor(header_np, cv2.COLOR_RGB2BGR)  # Correct color conversion

    # --- Answer Section Selection ---
    st.write("Select the Answer Section")
    cropped_answers = st_cropper(
        image,
        realtime_update=True,
        box_color='#FF0000',  # Red color for answer box
        aspect_ratio=(4, 3),  # No aspect ratio constraint
    )
    # Convert cropped image to OpenCV format
    answers_np = np.array(cropped_answers)
    answers_np = cv2.cvtColor(answers_np, cv2.COLOR_RGB2BGR)  # Correct color conversion

    # --- Divide Answer Section Vertically ---
    answer_width = answers_np.shape[1] // 4
    answer_sections = []
    for i in range(4):
        start_x = i * answer_width
        end_x = (i + 1) * answer_width
        answer_section = answers_np[:, start_x:end_x]  # Slice vertically
        answer_sections.append(answer_section)


    # --- Display Final Extracted Sections ---
    st.subheader("Final Sections")
    st.image(header_np, caption="Final Header Section", use_container_width=True)
    st.image(answers_np, caption="Final Answer Section", use_container_width=True)

    # Display the original answer section and the divided sections
    st.subheader("Divided Answer Sections")
    for i, answer_section in enumerate(answer_sections):
        st.image(answer_section, caption=f"Answer Section {i+1}", use_container_width=True)

    # --- Debugging --- 
    st.subheader("Debugging")
    st.write("Thresholded Image Result")
    thresh_img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    st.image(thresh_img_rgb, caption="Thresholded Image Result", use_container_width=True)