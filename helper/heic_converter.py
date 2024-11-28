# heic_converter.py
import os
import logging
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
import tempfile

logging.basicConfig(level=logging.INFO, format='%(message)s')

def convert_single_file_to_path(heic_path, output_quality=90):
    """
    Converts a single HEIC file to JPG format.

    Args:
        heic_path (str): Path to the HEIC file.
        output_quality (int): Quality of the output JPG image.

    Returns:
        str: Path to the converted JPG file.
    """
    register_heif_opener()

    if not os.path.isfile(heic_path):
        logging.error(f"File '{heic_path}' does not exist.")
        return None

    try:
        jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
        with Image.open(heic_path) as image:
            image.save(jpg_path, "JPEG", quality=output_quality)
        logging.info(f"Successfully converted '{heic_path}' to '{jpg_path}'.")
        return jpg_path
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error(f"Error converting '{heic_path}': {e}")
        return None
    
def convert_single_filePath_to_img_obj(heic_path, output_quality=90):
    """
    Converts a single HEIC file to a JPG image object.

    Args:
        heic_path (str): Path to the HEIC file.
        output_quality (int): Quality of the output JPG image.

    Returns:
        Image: The converted JPG image object.
    """
    register_heif_opener()

    if not os.path.isfile(heic_path):
        logging.error(f"File '{heic_path}' does not exist.")
        return None

    try:
        with Image.open(heic_path) as image:
            image = image.convert("RGB")  # Ensure it's in RGB mode
            output_image = Image.new("RGB", image.size)
            output_image.paste(image)
            logging.info(f"Successfully converted '{heic_path}' to JPG format.")
            return output_image
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error(f"Error converting '{heic_path}': {e}")
        return None

def convert_single_fileBytes_to_img_obj(heic_file_bytes, output_quality=90):
    """
    Converts HEIC file bytes to a JPG image object.

    Args:
        heic_file_bytes (bytes): Bytes of the HEIC file.
        output_quality (int): Quality of the output JPG image.

    Returns:
        Image: The converted JPG image object.
    """
    register_heif_opener()

    try:
        with Image.open(heic_file_bytes) as image:
            image = image.convert("RGB")  # Ensure it's in RGB mode
            output_image = Image.new("RGB", image.size)
            output_image.paste(image)
            return output_image
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error(f"Error converting HEIC bytes: {e}")
        return None

def handle_uploaded_file(uploaded_file):
    """
    Handles the uploaded file by saving it to a temporary location, printing the path,
    and returning the path for further processing.
    
    Parameters:
        uploaded_file: Uploaded file object from Streamlit file uploader.

    Returns:
        str: Path to the temporary file.
    """
    if uploaded_file is not None:
        # Create a temporary file and save uploaded file contents
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Print and return the temporary file path for debugging
        print(f"Temporary file path: {temp_file_path}")
        return temp_file_path
    else:
        print("ERROR: No file uploaded!")
        return None


if __name__ == "__main__":
    # Test case
    test_heic_file = "D:\\156.HEIC"  # Replace with your HEIC file path
    output_quality = 90

    logging.info("Starting HEIC to JPG conversion test...")
    jpg_image = convert_single_filePath_to_img_obj(test_heic_file, output_quality)

    if jpg_image:
        # Display the image directly without saving it
        jpg_image.show()  # Opens the image using the default system viewer
        logging.info("Displayed the converted image directly.")
    else:
        logging.error("Test failed.")

