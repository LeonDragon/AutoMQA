# TODO
# STAGE 1:
#   FIRST: Try adjust perspective of image
#   SECOND: Crop the answer sheet based on countour detected in first step
# STAGE 2:

import streamlit as st
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from streamlit_cropper import st_cropper
from PIL import Image  # Import Pillow library

st.title("Answer Area Detection with Bubble Detection")

# File upload
uploaded_file = st.file_uploader("Upload an answer sheet image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image_org = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1) 
    image_stage1_org= image_org.copy()

    # --- Adjust Perspective using Vertical Lines ---
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)

    # Calculate median angle and ROTATE image (with angle restriction)
    median_angle = np.median(angles)
    if abs(median_angle) > 45:  # Check if angle exceeds 45 degrees
        st.warning("Detected rotation angle exceeds 45 degrees. Skipping automatic rotation.")
    else:
        rows, cols = image_org.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
        image_stage1_org = cv2.warpAffine(image_org, M, (cols, rows))

    image_stage1_proc = image_stage1_org.copy()

    # --- End of Perspective Adjustment ---

    # Manual cropping with confirmation (Fixed)
    if "manual_crop_active" not in st.session_state:
        st.session_state.manual_crop_active = False

    if st.button("Crop Image Manually"):
        st.session_state.manual_crop_active = True  # Activate cropping mode

    if st.session_state.manual_crop_active:
        pil_image = Image.fromarray(image_stage1_org)

        if "cropped_img" not in st.session_state:
            st.session_state.cropped_img = None

        cropped_img_temp = st_cropper(
            pil_image,
            realtime_update=True,
            box_color="green",
            aspect_ratio=None,
            key="cropper_1"
        )

        if st.button("Confirm Crop"):
            if cropped_img_temp is not None:
                st.session_state.cropped_img = cropped_img_temp
                st.session_state.manual_crop_active = False  # Deactivate cropping mode
            else:
                st.warning("Please select a cropping region first.")

        if st.session_state.cropped_img is not None:
            image_stage1_org = cv2.cvtColor(np.array(st.session_state.cropped_img), cv2.COLOR_RGB2BGR)
            st.image(image_stage1_org, caption="Manually Cropped Image", use_column_width=True)

    image_stage1_proc = image_stage1_org.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image_stage1_proc, cv2.COLOR_BGR2GRAY)

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
        if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
            bubble_coords.append((x, y, w, h))
            cv2.rectangle(image_stage1_proc, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Draw for visualization

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
            padding = 10
            x_min -= padding + 80
            y_min -= padding
            x_max += padding
            y_max += padding

            # Extract answer area using perspective transform
            answer_area_contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            warped = four_point_transform(image_stage1_org, answer_area_contour.reshape(4, 2))

            cv2.rectangle(image_stage1_proc, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw answer area
            st.image(image_stage1_proc, caption="Detected Bubbles and Answer Area", use_container_width=True)
            st.image(warped, caption="Answer Area after Perspective Transform", use_container_width=True)

            # Divide warped image into 4 columns
            warped_height, warped_width = warped.shape[:2]
            column_width = warped_width // 4
            columns = [
                warped[0:warped_height, 0:column_width],
                warped[0:warped_height, column_width:2 * column_width],
                warped[0:warped_height, 2 * column_width:3 * column_width],
                warped[0:warped_height, 3 * column_width:warped_width]
            ]

            # Display the 4 columns
            for i, col in enumerate(columns):
                st.image(col, caption=f"Column {i+1}", use_container_width=True)

            # Extract header (from top edge to y_min, full width)
            header = image_stage1_org[0:y_min, 0:image_stage1_org.shape[1]]  # Corrected slicing
            st.image(header, caption="Answer Sheet Header", use_container_width=True)

        else:
            st.warning("No bubbles with similar size found.")
    else:
        st.warning("No bubbles found.")

    # Debug displays (optional)
    st.image(gray, caption="Grayscale Image", use_container_width=True)
    st.image(blurred, caption="Blurred Image", use_container_width=True)
    st.image(thresh, caption="Adaptive Thresholding", use_container_width=True)