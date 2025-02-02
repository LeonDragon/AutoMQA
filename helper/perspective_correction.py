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

def adjust_perspective_column_image(img_np):
    """
    Deskews the input image (column image) by:
      1. Converting to grayscale (if not already) and blurring.
      2. Thresholding (using Otsu's method) to produce a binary image.
      3. Applying a morphological closing operation to merge nearby numbers/bubbles.
      4. Finding the largest contour (assumed to contain the numbers and bubbles).
      5. Computing the minimum-area bounding rectangle for that contour.
      6. Extracting its angle and rotating the image accordingly.
      
    The function returns the deskewed image as a numpy array.
    """
    # --- Step 1: Preprocessing ---
    # If image is already single channel, assume it's grayscale.
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        gray = img_np.copy()
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # --- Step 2: Thresholding ---
    # Use Otsu's method; THRESH_BINARY_INV makes numbers and bubbles white.
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("Otsu threshold value:", ret)
    
    # --- Step 3: Morphological Closing ---
    # A relatively large rectangular kernel to close gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # --- Step 4: Find the largest contour ---
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return img_np  # Return the original image if nothing is found.
    
    # Assume the largest contour is our block of numbers and bubbles.
    largest_contour = max(contours, key=cv2.contourArea)
    
    # --- Step 5: Compute minimum area rectangle ---
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    # Adjust the angle: if less than -45, add 90 degrees.
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    print("Detected angle:", angle)
    
    # # --- (Optional) Visualization: Draw the bounding rectangle ---
    # vis_img = img_np.copy()
    # cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)
    
    # plt.figure(figsize=(15, 10))
    
    # plt.subplot(2, 3, 1)
    # if len(img_np.shape) == 2:
    #     plt.imshow(img_np, cmap="gray")
    # else:
    #     plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 2)
    # plt.imshow(gray, cmap="gray")
    # plt.title("Grayscale")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 3)
    # plt.imshow(blurred, cmap="gray")
    # plt.title("Blurred")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 4)
    # plt.imshow(thresh, cmap="gray")
    # plt.title("Threshold (Otsu)")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 5)
    # plt.imshow(closed, cmap="gray")
    # plt.title("Morphologically Closed")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 6)
    # if len(vis_img.shape) == 2:
    #     plt.imshow(vis_img, cmap="gray")
    # else:
    #     plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    # plt.title("Bounding Rectangle")
    # plt.axis("off")
    
    # plt.tight_layout()
    # plt.show()
    
    # --- Step 6: Rotate the image to deskew ---
    if abs(angle) > 45:
        print("Angle exceeds ±45°. Skipping rotation.")
        return img_np  # Skip rotation and return the original image.
    
    (h, w) = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # # (Optional) Visualization: Show the rotated image.
    # plt.figure(figsize=(6, 6))
    # if len(rotated_img.shape) == 2:
    #     plt.imshow(rotated_img, cmap="gray")
    # else:
    #     plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    # plt.title("Rotated (Deskewed) Image")
    # plt.axis("off")
    # plt.show()
    
    return rotated_img

# --------------------------
# Example usage (e.g., in Google Colab with file upload):
# --------------------------
# from google.colab import files
# uploaded = files.upload()
# if uploaded:
#     for fn in uploaded.keys():
#         print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
#         # Read image from uploaded file
#         img_np = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)
#         if img_np is None:
#             print("Could not load image. Check the file format.")
#         else:
#             deskewed_img = adjust_perspective_column_image(img_np)

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