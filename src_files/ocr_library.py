# ocr_library.py

from PIL import Image
import pytesseract
import re

def extract_data_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(img)

    # Define a dictionary to store the extracted data
    data = {}

    # Split the text into lines
    lines = text.split('\n')

    # Regular expression to match key-value pairs
    key_value_pattern = re.compile(r'([A-Za-z\s]+):\s*([\d.]+ x \d+/\w+|\d+\s*\w+/\w+|\d+.\d+|\d+)')

    # Iterate through the lines and extract key-value pairs
    for line in lines:
        match = key_value_pattern.match(line)
        if match:
            key, value = match.groups()
            data[key.strip()] = value.strip()

    return data
