import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from gemini_utils import process_answer_key, process_student_answers, process_single_column, recheck_single_column
from helper.heic_converter import convert_single_fileBytes_to_img_obj
from helper.perspective_correction import adjust_perspective
from helper.est_answer_area import infer_answer_area_average_size
from imutils.perspective import four_point_transform

# Set page config
st.set_page_config(
    page_title="OMR Sheet Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .upload-section {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_answer_keys():
    """Load answer keys from JSON file"""
    try:
        with open('answer_keys.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading answer keys: {str(e)}")
        return None

def process_image(image_data, settings):
    """Process the uploaded image with given settings"""
    try:
        # Convert image data to numpy array
        if isinstance(image_data, bytes):
            img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            img_np = image_data
            
        if img_np is None:
            st.error("Failed to decode image")
            return None
            
        # Apply perspective correction
        img_np = adjust_perspective(img_np)
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(denoised)
        
        # Process bubbles
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_coords = []
        
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            
            if (w >= settings['min_width'] and 
                h >= settings['min_height'] and 
                ar >= settings['min_aspect_ratio'] and 
                ar <= settings['max_aspect_ratio']):
                
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.5:
                    bubble_coords.append((x, y, w, h))
                    cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Process answer area
        if bubble_coords:
            x_min, y_min, x_max, y_max = infer_answer_area_average_size(bubble_coords)
            if x_min is not None:
                padding = 15
                x_min -= padding + 90
                y_min -= padding
                x_max += padding
                y_max += padding
                
                answer_area_contour = np.array([
                    [x_min, y_min], 
                    [x_max, y_min],
                    [x_max, y_max], 
                    [x_min, y_max]
                ])
                
                warped = four_point_transform(
                    enhanced_gray.copy(), 
                    answer_area_contour.reshape(4, 2)
                )
                
                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Split into columns and vertical groups
                warped_height, warped_width = warped.shape[:2]
                column_width = warped_width // 4
                columns = []
                vertical_groups = []
                
                for i in range(4):
                    col = warped[0:warped_height, i*column_width:(i+1)*column_width]
                    columns.append(col)
                    
                    col_height = col.shape[0]
                    row_height = col_height // 3
                    for j in range(3):
                        row_group = col[j*row_height:(j+1)*row_height, 0:column_width]
                        _, row_buffer = cv2.imencode('.jpg', row_group)
                        vertical_groups.append({
                            'column': i,
                            'group': j,
                            'image': row_buffer
                        })
                
                header = img_np[0:y_min, 0:img_np.shape[1]]
                
                return {
                    'success': True,
                    'processed_image': img_np,
                    'warped_image': warped,
                    'columns': columns,
                    'header': header,
                    'vertical_groups': vertical_groups
                }
        
        return None
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'min_width': 30,
            'min_height': 5,
            'min_aspect_ratio': 0.9,
            'max_aspect_ratio': 1.2
        }
    
    # Load answer keys from JSON
    try:
        with open('answer_keys.json', 'r') as f:
            st.session_state.answer_key_data = json.load(f)['answerKeys']
    except Exception as e:
        st.error(f"Error loading answer keys: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.header("Processing Settings")
        st.session_state.settings['min_width'] = st.slider(
            "Min Width", 20, 100, 
            st.session_state.settings['min_width']
        )
        st.session_state.settings['min_height'] = st.slider(
            "Min Height", 1, 50, 
            st.session_state.settings['min_height']
        )
        st.session_state.settings['min_aspect_ratio'] = st.slider(
            "Min Aspect Ratio", 0.5, 1.0, 
            st.session_state.settings['min_aspect_ratio']
        )
        st.session_state.settings['max_aspect_ratio'] = st.slider(
            "Max Aspect Ratio", 1.0, 1.5, 
            st.session_state.settings['max_aspect_ratio']
        )

    # Main content
    st.title("OMR Sheet Analyzer")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Answer Keys", "📄 Student Sheet", "📊 Results"])
    
    # Tab 1: Answer Keys Display
    with tab1:
        st.header("Answer Keys")
        if st.session_state.answer_key_data:
            st.json(st.session_state.answer_key_data)
        else:
            st.error("No answer keys loaded. Please check answer_keys.json file.")
    
    # Tab 2: Student Sheet Processing
    with tab2:
        st.header("Process Student Sheet")
        
        student_sheet = st.file_uploader(
            "Choose a student answer sheet",
            type=['jpg', 'jpeg', 'png', 'heic'],
            key="student_sheet"
        )
        
        if student_sheet:
            try:
                if student_sheet.name.lower().endswith('.heic'):
                    img_obj = convert_single_fileBytes_to_img_obj(student_sheet.read())
                    if img_obj is None:
                        st.error("Failed to convert HEIC file")
                    else:
                        image_data = cv2.cvtColor(np.array(img_obj), cv2.COLOR_RGB2BGR)
                else:
                    file_bytes = student_sheet.read()
                    image_data = cv2.imdecode(
                        np.frombuffer(file_bytes, np.uint8), 
                        cv2.IMREAD_COLOR
                    )
                
                st.image(image_data, caption="Student Sheet Preview", use_column_width=True)
                
                if st.button("Process Sheet"):
                    with st.spinner("Processing student sheet..."):
                        results = process_image(image_data, st.session_state.settings)
                        
                        if results and results['success']:
                            st.session_state.processing_results = results
                            
                            # Display processed images
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Processed Image")
                                st.image(results['processed_image'], use_column_width=True)
                            with col2:
                                st.subheader("Warped Image")
                                st.image(results['warped_image'], use_column_width=True)
                            
                            # Display columns
                            st.subheader("Answer Columns")
                            cols = st.columns(4)
                            for idx, col_data in enumerate(results['columns']):
                                with cols[idx]:
                                    st.image(
                                        col_data,
                                        caption=f"Column {idx + 1}",
                                        use_column_width=True
                                    )
                        else:
                            st.error("Failed to process student sheet")
            
            except Exception as e:
                st.error(f"Error processing student sheet: {str(e)}")
    
    # Tab 3: Results and Analysis
    with tab3:
        st.header("Analysis Results")
        
        if st.session_state.processing_results and st.session_state.answer_key_data:
            model_name = st.selectbox(
                "Select Gemini Model",
                ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
            )
            
            if st.button("Analyze with Gemini"):
                with st.spinner("Processing with Gemini..."):
                    try:
                        # Process vertical groups
                        answer_key_data, answers, scores, responses, token_usage = process_student_answers(
                            st.session_state.processing_results['vertical_groups'],
                            model_name,
                            "answer_keys.json"
                        )
                        
                        if answers:
                            st.success("Analysis complete!")
                            
                            # Display token usage
                            st.info(f"""
                                Token Usage:
                                - Input: {token_usage['input_tokens']}
                                - Output: {token_usage['output_tokens']}
                            """)
                            
                            # Display scores
                            st.subheader("Scores")
                            for exam_code, score in scores.items():
                                st.metric(f"Exam {exam_code}", f"{score:.2f}%")
                            
                            # Display answers
                            st.subheader("Detected Answers")
                            st.json(answers)
                            
                            # Display raw responses
                            with st.expander("Show Raw Responses"):
                                for idx, response in enumerate(responses):
                                    st.text(f"Column {idx + 1}:")
                                    st.code(response)
                        else:
                            st.error("Failed to analyze answers")
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please process both answer key and student sheet first")

if __name__ == "__main__":
    main()
import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import cv2
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="OMR Sheet Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .upload-section {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Flask API endpoint
FLASK_API = "http://localhost:5000"

def process_answer_key(file):
    """Send answer key to Flask API for processing"""
    try:
        files = {'answer_key': file}
        response = requests.post(f"{FLASK_API}/process_answer_key", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error processing answer key: {str(e)}")
        return None

def process_student_sheet(file, settings):
    """Send student sheet to Flask API for processing"""
    try:
        files = {'student_sheet': file}
        response = requests.post(
            f"{FLASK_API}/process_student_sheet",
            files=files,
            data=settings
        )
        return response.json()
    except Exception as e:
        st.error(f"Error processing student sheet: {str(e)}")
        return None

def process_single_column(column_data, model_name, column_index, is_recheck=False):
    """Process a single column with Gemini"""
    try:
        payload = {
            'model_name': model_name,
            'column': column_data,
            'index': column_index,
            'recheck': is_recheck
        }
        response = requests.post(
            f"{FLASK_API}/process_single_column",
            json=payload
        )
        return response.json()
    except Exception as e:
        st.error(f"Error processing column: {str(e)}")
        return None

def process_all_columns(vertical_groups, model_name):
    """Process all columns with Gemini"""
    try:
        payload = {
            'model_name': model_name,
            'vertical_groups': vertical_groups
        }
        response = requests.post(
            f"{FLASK_API}/process_all_columns",
            json=payload
        )
        return response.json()
    except Exception as e:
        st.error(f"Error processing columns: {str(e)}")
        return None

def process_with_gemini_legacy(columns, model_name):
    """Legacy endpoint for Gemini processing"""
    try:
        payload = {
            'model_name': model_name,
            'columns': columns
        }
        response = requests.post(
            f"{FLASK_API}/process_with_gemini",
            json=payload
        )
        return response.json()
    except Exception as e:
        st.error(f"Error in Gemini processing: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'min_width': 30,
            'min_height': 5,
            'min_aspect_ratio': 0.9,
            'max_aspect_ratio': 1.2
        }
    
    # Load answer keys from JSON
    try:
        with open('answer_keys.json', 'r') as f:
            st.session_state.answer_key_data = json.load(f)['answerKeys']
    except Exception as e:
        st.error(f"Error loading answer keys: {str(e)}")

    # Sidebar settings
    with st.sidebar:
        st.header("Processing Settings")
        st.session_state.settings['min_width'] = st.slider(
            "Min Width", 20, 100, 
            st.session_state.settings['min_width']
        )
        st.session_state.settings['min_height'] = st.slider(
            "Min Height", 1, 50, 
            st.session_state.settings['min_height']
        )
        st.session_state.settings['min_aspect_ratio'] = st.slider(
            "Min Aspect Ratio", 0.5, 1.0, 
            st.session_state.settings['min_aspect_ratio']
        )
        st.session_state.settings['max_aspect_ratio'] = st.slider(
            "Max Aspect Ratio", 1.0, 1.5, 
            st.session_state.settings['max_aspect_ratio']
        )

    # Main content
    st.title("OMR Sheet Analyzer")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Answer Keys", "📄 Student Sheet", "📊 Results"])
    
    # Tab 1: Answer Keys Display
    with tab1:
        st.header("Answer Keys")
        if st.session_state.answer_key_data:
            st.json(st.session_state.answer_key_data)
        else:
            st.error("No answer keys loaded. Please check answer_keys.json file.")
    
    # Tab 2: Student Sheet Processing
    with tab2:
        st.header("Process Student Sheet")
        
        uploaded_file = st.file_uploader(
            "Choose a student answer sheet",
            type=['jpg', 'jpeg', 'png', 'heic']
        )
        
        if uploaded_file:
            # Display preview
            st.image(uploaded_file, caption="Preview", use_column_width=True)
            
            if st.button("Process Sheet"):
                with st.spinner("Processing..."):
                    # Process the sheet
                    results = process_student_sheet(uploaded_file, st.session_state.settings)
                    
                    if results and results.get('success'):
                        st.session_state.processing_results = results
                        
                        # Display processed images
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Processed Image")
                            st.image(
                                base64.b64decode(results['processed_image']),
                                use_column_width=True
                            )
                        with col2:
                            st.subheader("Warped Image")
                            st.image(
                                base64.b64decode(results['warped_image']),
                                use_column_width=True
                            )
                        
                        # Display columns
                        st.subheader("Answer Columns")
                        cols = st.columns(4)
                        for idx, col_data in enumerate(results['columns']):
                            with cols[idx]:
                                st.image(
                                    base64.b64decode(col_data),
                                    caption=f"Column {idx + 1}",
                                    use_column_width=True
                                )
                    else:
                        st.error("Failed to process student sheet")
    
    # Tab 3: Results and Analysis
    with tab3:
        st.header("Analysis Results")
        
        if st.session_state.processing_results and st.session_state.answer_key_data:
            model_name = st.selectbox(
                "Select Gemini Model",
                ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process All Columns"):
                    with st.spinner("Processing all columns..."):
                        results = process_all_columns(
                            st.session_state.processing_results['vertical_groups'],
                            model_name
                        )
                        
                        if results and results.get('success'):
                            # Display scores
                            st.subheader("Scores")
                            for exam_code, score in results.get('scores', {}).items():
                                st.metric(f"Exam {exam_code}", f"{score:.2f}%")
                            
                            # Display detected answers
                            st.subheader("Detected Answers")
                            st.json(results.get('answers', {}))
                            
                            # Display raw responses
                            with st.expander("Show Raw Responses"):
                                for idx, response in enumerate(results.get('all_responses', [])):
                                    st.text(f"Column {idx + 1}:")
                                    st.code(response)
                    else:
                        st.error("Failed to analyze answers")
        else:
            st.warning("Please process both answer key and student sheet first")

if __name__ == "__main__":
    main()
