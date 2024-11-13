import cv2
import os
import easyocr
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import re
from difflib import SequenceMatcher

def extract_screenshots(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # Initialize the frame counter
    frame_count = 0
    success = True

    while success:
        # Read a frame from the video
        success, frame = video.read()

        # Break the loop if no more frames are available
        if not success:
            print("Finished reading all frames.")
            break

        # Check if the current frame is at a 1-second interval
        if frame_count % int(fps) == 0:
            # Generate the output filename
            output_filename = f"screenshot_{frame_count // int(fps):04d}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # Save the frame as an image file
            cv2.imwrite(output_path, frame)
            print(f"Saved screenshot: {output_path}")

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    video.release()
    print("Released video capture.")


def clean_text(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace incorrect punctuation and add full stops where appropriate
    text = re.sub(r'[;:]', '.', text)
    return text


def find_subtitle_region(image, reader):
    # Divide the image into 3 horizontal regions: top, center, bottom
    width, height = image.size
    top_region = image.crop((0, 0, width, height // 3))
    center_region = image.crop((0, height // 3, width, 2 * height // 3))
    bottom_region = image.crop((0, 2 * height // 3, width, height))

    # Perform OCR on each region
    top_result = reader.readtext(np.array(top_region))
    center_result = reader.readtext(np.array(center_region))
    bottom_result = reader.readtext(np.array(bottom_region))

    # Count the number of text detections in each region
    top_text_count = len(top_result)
    center_text_count = len(center_result)
    bottom_text_count = len(bottom_result)

    # Determine the region with the most text detections
    if top_text_count >= center_text_count and top_text_count >= bottom_text_count:
        return top_region
    elif center_text_count >= top_text_count and center_text_count >= bottom_text_count:
        return center_region
    else:
        return bottom_region


def similar(a, b, threshold=0.7):
    # Use SequenceMatcher to compare two strings
    return SequenceMatcher(None, a, b).ratio() >= threshold


def extract_text_from_images(image_dir, output_txt_path):
    print(f"Started extracting subtitles...")

    # Store the detected subtitles
    subtitles = []

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # List all image files in the directory
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    last_subtitles = []  # List to keep track of the last few subtitles
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Convert the image to greyscale
        greyscale_image = image.convert('L')

        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(greyscale_image)
        greyscale_image = contrast_enhancer.enhance(1.5)  # Increase contrast

        # Invert colors to make text white and background dark
        inverted_image = ImageOps.invert(greyscale_image)

        # Find the subtitle region
        subtitle_region = find_subtitle_region(inverted_image, reader)

        # Perform OCR on the subtitle region
        result = reader.readtext(np.array(subtitle_region))

        # Extract text from the result
        text = ' '.join([entry[1] for entry in result])

        # Clean the detected text
        cleaned_text = clean_text(text)

        # Check for duplicates and similarity
        if cleaned_text:
            if len(last_subtitles) < 5:  # Keep track of the last 5 subtitles
                last_subtitles.append(cleaned_text)
            else:
                last_subtitles.pop(0)
                last_subtitles.append(cleaned_text)

            # Check if the cleaned text is similar to any of the last few subtitles
            if all(not similar(cleaned_text, subtitle) for subtitle in last_subtitles[:-1]):
                subtitles.append(cleaned_text)
                print(f"Added subtitle: {cleaned_text}")

    # Write the unique subtitles to a text file
    with open(output_txt_path, 'w') as f:
        for subtitle in subtitles:
            f.write(subtitle + "\n")  # Add a newline for readability

    print("Finished extracting subtitles.")


# Usage example
# Make sure this path is correct
video_path = "C:/Users/zubai/Desktop/New folder/Try.mov"
output_dir = "./output_screenshots"
output_txt_path = "./output_subtitles.txt"

extract_screenshots(video_path, output_dir)
extract_text_from_images(output_dir, output_txt_path)
