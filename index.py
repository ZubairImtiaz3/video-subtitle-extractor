import cv2
import os
import easyocr
from PIL import Image, ImageEnhance, ImageOps
import numpy as np


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


def extract_text_from_images(image_dir, output_txt_path):
    print(f"Started extracting subtitles...")

    # Store the detected subtitles
    subtitles = []

    # List all image files in the directory
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Crop the bottom half of the image
        width, height = image.size
        cropped_image = image.crop((0, height - 120, width, height))

        # Convert cropped image to greyscale
        greyscale_image = cropped_image.convert('L')

        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(greyscale_image)
        greyscale_image = contrast_enhancer.enhance(1.5)  # Increase contrast

        # Invert colors to make text white and background dark
        inverted_image = ImageOps.invert(greyscale_image)

        # Convert the PIL image to a numpy array
        inverted_image_np = np.array(inverted_image)

       # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Perform OCR on inverted image
        result = reader.readtext(inverted_image_np)

        # Extract text from the result
        text = ' '.join([entry[1] for entry in result])

        # Clean and store the detected text
        cleaned_text = text.strip()
        if cleaned_text and (len(subtitles) == 0 or cleaned_text != subtitles[-1]):
            subtitles.append(cleaned_text)

    # Write the unique subtitles to a text file
    with open(output_txt_path, 'w') as f:
        for subtitle in subtitles:
            f.write(subtitle + "\n")

    print("Finished extracting subtitles.")


# Usage example
# Make sure this path is correct
video_path = "C:/Users/zubai/Desktop/New folder/Try.mov"
output_dir = "./output_screenshots"
output_txt_path = "./output_subtitles.txt"

extract_screenshots(video_path, output_dir)
extract_text_from_images(output_dir, output_txt_path)
