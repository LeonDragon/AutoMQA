# preprocessing.py
import cv2
import numpy as np
import math
from imutils.perspective import four_point_transform

def preprocess_image(image):
    """
    Preprocesses the input image by performing perspective correction and cropping.

    Args:
        image: A NumPy array representing the input image.

    Returns:
        A tuple containing:
            - processed_image: The preprocessed image as a NumPy array.
            - warped_image: The warped image after perspective transform (if successful).
            - columns: A list of 4 NumPy arrays representing the divided answer columns.
            - header: A NumPy array representing the extracted header.
    """

    img_np = np.array(image)  # Convert PIL Image to NumPy array

    # --- Adjust Perspective ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)

    # Calculate median angle and ROTATE image (with angle restriction)
    if angles:
        median_angle = np.median(angles)
        if abs(median_angle) > 45:  # Check if angle exceeds 45 degrees
            print("Detected rotation angle exceeds 45 degrees. Skipping automatic rotation.")
        else:
            rows, cols = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
            img_np = cv2.warpAffine(img_np, M, (cols, rows))
    else:
        print("No lines detected for perspective correction.")

    # --- Bubble Detection and Answer Area Extraction ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_coords = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
            bubble_coords.append((x, y, w, h))

    warped_image = None  # Initialize warped_image
    columns = None  # Initialize columns
    header = None  # Initialize header

    if bubble_coords:
        avg_width = sum(w for _, _, w, _ in bubble_coords) / len(bubble_coords)
        avg_height = sum(h for _, _, _, h in bubble_coords) / len(bubble_coords)

        filtered_coords = [(x, y, w, h) for x, y, w, h in bubble_coords
                           if abs(w - avg_width) < 0.2 * avg_width and abs(h - avg_height) < 0.2 * avg_height]

        if filtered_coords:
            x_min = min(x for x, _, _, _ in filtered_coords)
            y_min = min(y for _, y, _, _ in filtered_coords)
            x_max = max(x + w for x, _, w, _ in filtered_coords)
            y_max = max(y + h for _, y, _, h in filtered_coords)

            padding = 10
            x_min -= padding + 80
            y_min -= padding
            x_max += padding
            y_max += padding

            answer_area_contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            warped_image = four_point_transform(img_np.copy(), answer_area_contour.reshape(4, 2))

            # Divide warped image into 4 columns
            warped_height, warped_width = warped_image.shape[:2]
            column_width = warped_width // 4
            columns = [
                warped_image[0:warped_height, 0:column_width],
                warped_image[0:warped_height, column_width:2 * column_width],
                warped_image[0:warped_height, 2 * column_width:3 * column_width],
                warped_image[0:warped_height, 3 * column_width:warped_width]
            ]

            # Extract header
            header = img_np[0:y_min, 0:img_np.shape[1]]

    return img_np, warped_image, columns, header

def enhance_bubbles(image_np, threshold=120):
    """
    Enhance filled bubbles by drawing black circles on them using thresholding and morphological operations.
    
    Args:
        image_np (numpy.ndarray): Input image (can be grayscale or BGR)
        threshold (int): Threshold value for binary inverse thresholding
        
    Returns:
        numpy.ndarray: Processed image with enhanced bubbles
    """
    # Convert to grayscale if needed
    if len(image_np.shape) == 3:  # If image has 3 channels (BGR)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:  # If already grayscale
        gray = image_np
    
    # Apply binary inverse thresholding
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw on
    output = image_np.copy()
    
    # Process each contour
    for contour in contours:
        # Calculate moments and centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate radius
        area = cv2.contourArea(contour)
        radius = int(math.sqrt(area / math.pi))
        
        # Filter by size (optional)
        if 5 < radius < 50:  # Adjust these values as needed
            # Draw black circle
            cv2.circle(output, (cx, cy), radius, (0, 0, 0), -1)
    
    return output
