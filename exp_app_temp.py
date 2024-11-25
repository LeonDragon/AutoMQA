import streamlit as st
import cv2
import numpy as np

st.title("Answer Area Detection with Contour Detection")

# File upload
uploaded_file = st.file_uploader("Upload an answer sheet image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    original_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    # Dilation to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Canny edge detection
    edges = cv2.Canny(dilated, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate minimum area based on image size
    image_area = image.shape[0] * image.shape[1]
    min_area = 0.2 * image_area  # Adjusted based on testing

    answer_sheet_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        # Check if the contour has 4 sides and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            answer_sheet_contour = approx
            break

    # Draw the contour on the original image
    if answer_sheet_contour is not None:
        cv2.drawContours(original_image, [answer_sheet_contour], -1, (0, 255, 0), 2)
        st.image(original_image, caption="Detected Answer Area", use_column_width=True)
    else:
        st.warning("Answer sheet area not found.")

    # Debug displays
    st.image(gray, caption="Grayscale Image", use_column_width=True)
    st.image(blurred, caption="Blurred Image", use_column_width=True)
    st.image(thresh, caption="Adaptive Thresholding", use_column_width=True)
    st.image(dilated, caption="Dilation", use_column_width=True)
    st.image(edges, caption="Canny Edges", use_column_width=True)