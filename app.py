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
from sklearn.cluster import DBSCAN  # Import DBSCAN
from gemini_utils import upload_to_gemini, process_answer_key, process_student_answers  # Import from gemini_utils.py


# --- Row/Column Analysis ---
def infer_answer_area_row_col(bubble_coords):
    """Infers the answer area using row/column analysis."""

    if not bubble_coords:
        return None

    # Sort by y-coordinate (rows)
    bubble_coords.sort(key=lambda coord: coord[1])  # Sort by y

    rows = []
    current_row = []
    last_y = bubble_coords[0][1]
    row_threshold = 20  # Adjust this threshold based on bubble spacing

    for x, y, w, h in bubble_coords:
        if abs(y - last_y) > row_threshold:
            # Start a new row
            rows.append(current_row)
            current_row = []
        current_row.append((x, y, w, h))
        last_y = y
    rows.append(current_row)  # Add the last row

    # Sort within each row by x-coordinate (columns)
    for row in rows:
        row.sort(key=lambda coord: coord[0])  # Sort by x

    # (Optional) Analyze distances between bubbles to refine row/column detection
    # ...

    # Define answer area using outermost bubbles
    x_min = min(row[0][0] for row in rows)
    y_min = rows[0][0][1]
    x_max = max(row[-1][0] + row[-1][2] for row in rows)
    y_max = rows[-1][0][1] + rows[-1][0][3]

    # Add some padding
    padding = 15
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    return x_min, y_min, x_max, y_max

# --- Expanding Bounding Box ---
def infer_answer_area_expanding_box(bubble_coords):
    """Infers the answer area using an expanding bounding box."""

    if not bubble_coords:
        return None

    # Select initial bubbles
    bubble_coords.sort(key=lambda coord: (coord[0], coord[1]))  # Sort by x, then y
    top_left = bubble_coords[0]
    bottom_right = bubble_coords[-1]

    # Initial bounding box
    x_min, y_min = top_left[0], top_left[1]
    x_max, y_max = bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]

    expansion_increment = 10  # Adjust this increment
    max_iterations = 50  # Adjust this limit
    previous_count = 0
    stable_count = 0
    stability_threshold = 3  # Adjust this threshold

    for _ in range(max_iterations):
        bubble_count = 0
        for x, y, w, h in bubble_coords:
            if x_min <= x and y_min <= y and x + w <= x_max and y + h <= y_max:
                bubble_count += 1

        if bubble_count == previous_count:
            stable_count += 1
        else:
            stable_count = 0
        previous_count = bubble_count

        if stable_count >= stability_threshold:
            break

        # Expand the box
        x_min -= expansion_increment
        y_min -= expansion_increment
        x_max += expansion_increment
        y_max += expansion_increment

    return x_min, y_min, x_max, y_max

# --- Density-Based Clustering ---
def infer_answer_area_dbscan(bubble_coords):
    """Infers the answer area using DBSCAN clustering."""
    print(bubble_coords)
    if not bubble_coords:
        return None

    # Prepare data for DBSCAN (x, y coordinates)
    coords = np.array([[x, y] for x, y, _, _ in bubble_coords])

    # Apply DBSCAN
    dbscan = DBSCAN(eps=100, min_samples=5)  # Adjust eps and min_samples
    clusters = dbscan.fit_predict(coords)

    print("Clusters:", clusters)  # Print the cluster assignments
    print("Cluster Counts:", np.bincount(clusters + 1))  # Print the size of each cluster


    # Identify the largest cluster (excluding noise)
    cluster_counts = np.bincount(clusters + 1)  # +1 to handle -1 (noise)
    largest_cluster = np.argmax(cluster_counts[1:])  # Exclude noise cluster from argmax

    # Get the points in the largest cluster
    cluster_points = coords[clusters == largest_cluster]

    if len(cluster_points) > 0:
        # Calculate bounding box
        x_min = int(np.min(cluster_points[:, 0]))
        y_min = int(np.min(cluster_points[:, 1]))
        x_max = int(np.max(cluster_points[:, 0]))
        y_max = int(np.max(cluster_points[:, 1]))

        return x_min, y_min, x_max, y_max

    else:
        return 0,0,0,0  # No cluster found
    
def infer_answer_area_original(bubble_coords):
    """Infers the answer area using the original average-based method."""

    if not bubble_coords:
        return None, None, None, None  # Return None for all values

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

        return x_min, y_min, x_max, y_max  # Return the coordinates

    else:
        return None, None, None, None  # Return None for all values


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
st.subheader("Upload Answer Key (Image)")
answer_key = None
answer_key_image = st.file_uploader("Choose an answer key image file", type=["jpg", "jpeg", "png"])
if answer_key_image:
    answer_key_image = np.array(Image.open(answer_key_image))
    answer_keys = process_answer_key(answer_key_image)

    if answer_keys:
        st.write("Answer Keys:")
        for test_code, answer_key in answer_keys.items():
            st.write(f"Test Code: {test_code}")
            st.write(answer_key)
    else:
        st.error("Failed to extract answer key.")

# --- Student Answer Sheet Processing ---
st.subheader("Upload Student Answer Sheet")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    img_np_org = img_np.copy()
    
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

        # Sliders in the sidebar
        st.sidebar.subheader("Bubble Detection Parameters")
        min_width = st.sidebar.slider("Minimum Width (pixels)", 20, 100, 30, help="Minimum width of the contour to be considered as a bubble.")
        min_height = st.sidebar.slider("Minimum Height (pixels)", 1, 50, 5, help="Minimum height of the contour to be considered as a bubble.")

        min_aspect_ratio = st.sidebar.slider("Minimum Aspect Ratio", 0.5, 1.0, 0.9, help="Minimum aspect ratio (width/height) of the contour to be considered as a bubble. A value closer to 1 means the contour is more circular. For example, 0.9 means the contour can be slightly elongated.")
        max_aspect_ratio = st.sidebar.slider("Maximum Aspect Ratio", 1.0, 1.5, 1.2, help="Maximum aspect ratio (width/height) of the contour to be considered as a bubble. A value closer to 1 means the contour is more circular. For example, 1.1 means the contour can be slightly wider than it is tall.")

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

        # Find contours (potential bubbles)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_coords = []  # Store bubble coordinates

        total_contours = 0  # Initialize a counter for contours


        for c in cnts:
            
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if (w > 30 and ar >=0.8 and ar <= 1.2):
                total_contours += 1  # Increment the counter for each contour
                print(f"Contour: width={w} (pixels), height={h} (pixels), aspect_ratio={ar} (w/h)") 

            # Filter for bubble-like shapes (adjust these parameters as needed)
            if (w >= min_width and h >= min_height) and ar >= min_aspect_ratio and ar <= max_aspect_ratio:
                bubble_coords.append((x, y, w, h))
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw for visualization
        
        print(f"Total number of contours found: {total_contours}")  # Print the total count

        # Infer answer area from bubble coordinates
        if bubble_coords:
            # Choose one of the methods:
            #x_min, y_min, x_max, y_max = infer_answer_area_row_col(bubble_coords)
            #x_min, y_min, x_max, y_max = infer_answer_area_expanding_box(bubble_coords)
            #x_min, y_min, x_max, y_max = infer_answer_area_dbscan(bubble_coords)
            x_min, y_min, x_max, y_max = infer_answer_area_original(bubble_coords)



            if x_min is not None:  # Check if answer area was found
                # --- Add some padding to the answer area ---
                padding = 15
                x_min -= padding + 90  # More padding on the left side (adjust as needed)
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
        model_name = st.selectbox("Select Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro-latest"])

        answer_key_path = "answer_keys.json"  # Or get the path from a file uploader

        if st.button("Process with Gemini"):
            all_extracted_answers, scores = process_student_answers(columns, model_name, answer_key_path)

            if all_extracted_answers is not None and scores is not None:
                st.write("Scores:")
                for test_code, score in scores.items():
                    if score is not None:
                        st.success(f"Test Code {test_code}: {score:.2f}%")
                    else:
                        st.error(f"Failed to calculate score for Test Code {test_code}")
                # st.write("Correct Answers:", sum(a == b for a, b in zip(all_extracted_answers, answer_key.values())))  # You might need to adjust this based on how you want to handle multiple answer keys


