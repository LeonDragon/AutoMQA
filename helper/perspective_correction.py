import cv2
import numpy as np
import streamlit as st

def adjust_perspective(img_np):
    """Adjusts the perspective of the image."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # Calculate angles of detected lines
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            angles.append(angle)

        # Filter out angles that are likely not part of the answer sheet
        filtered_angles = [angle for angle in angles if abs(angle) < 45]

        if filtered_angles:
            median_angle = np.median(filtered_angles)
            print(f"Rotate median angle: Angle={median_angle}")

            # Normalize the angle to ensure it lies within a reasonable range
            if median_angle > 45:
                median_angle -= 90
            elif median_angle < -45:
                median_angle += 90

            rows, cols = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
            img_np = cv2.warpAffine(img_np, M, (cols, rows))
        else:
            st.warning("No reliable lines detected for perspective correction.")
    else:
        st.warning("No lines detected for perspective correction.")

    return img_np


def adjust_perspective_crop_by_coordinates(img_np, left, top, width, height):
    """
    Automatically rotates the image based on the bounding box coordinates and crops the image.
    
    Parameters:
    - img_np: numpy array of the image
    - left: left coordinate of the bounding box
    - top: top coordinate of the bounding box
    - width: width of the bounding box
    - height: height of the bounding box
    
    Returns:
    - rotated_cropped_image: numpy array of the rotated and cropped image
    """
    # Calculate the center of the bounding box
    center_x = left + width // 2
    center_y = top + height // 2
    
    # Define the vertices of the bounding box
    P1 = (left, top)
    P2 = (left + width, top)
    P3 = (left + width, top + height)
    P4 = (left, top + height)
    
    # Calculate the vector of the reference edge (P1 to P2)
    v = (P2[0] - P1[0], P2[1] - P1[1])
    
    # Calculate the angle of the vector with respect to the horizontal axis
    angle = np.arctan2(v[1], v[0]) * 180. / np.pi
    
    # Normalize the angle to be within the range [-45°, 45°]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    
    print(f"Rotate angle: Angle={angle}")

    # Rotate the image
    rows, cols = img_np.shape[:2]
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    rotated_image = cv2.warpAffine(img_np, M, (cols, rows))
    
    # Crop the rotated image
    rotated_cropped_image = rotated_image[top:top+height, left:left+width]
    
    return rotated_cropped_image