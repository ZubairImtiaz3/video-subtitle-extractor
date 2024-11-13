import cv2
import os
import easyocr
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import re
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

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


def log_message(message, text_widget):
    text_widget.insert(tk.END, message + "\n")
    text_widget.see(tk.END)  # Scroll to the end
    text_widget.update_idletasks()


def extract_subtitles(video_path, output_dir, text_widget):
    try:
        output_screenshots_dir = os.path.join(output_dir, "output_screenshots")
        output_txt_path = os.path.join(output_dir, "output_subtitles.txt")

        log_message("Extracting screenshots...", text_widget)
        extract_screenshots(video_path, output_screenshots_dir)
        log_message("Screenshots extracted.", text_widget)

        log_message("Extracting text from images...", text_widget)
        extract_text_from_images(output_screenshots_dir, output_txt_path)
        log_message("Text extraction completed. Subtitles saved to output_subtitles.txt", text_widget)

    except Exception as e:
        log_message(f"Error: {str(e)}", text_widget)


def start_extraction(video_path, output_dir, text_widget):
    if not video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    if not output_dir:
        messagebox.showerror("Error", "Please select an output directory.")
        return

    # Start the extraction in a separate thread to keep the GUI responsive
    threading.Thread(target=extract_subtitles, args=(video_path, output_dir, text_widget)).start()


def create_gui():
    root = tk.Tk()
    root.title("Subtitle Extractor")

    # Set default output directory to desktop
    default_output_dir = os.path.join(os.path.expanduser("~"), "Desktop")

    # Video Path Selection
    video_label = tk.Label(root, text="Select Video:")
    video_label.pack(pady=5)

    video_entry = tk.Entry(root, width=50)
    video_entry.pack(pady=5)

    def browse_video():
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, file_path)

    browse_video_btn = tk.Button(root, text="Browse", command=browse_video)
    browse_video_btn.pack(pady=5)

    # Output Directory Selection
    output_label = tk.Label(root, text="Output Directory:")
    output_label.pack(pady=5)

    output_entry = tk.Entry(root, width=50)
    output_entry.insert(0, default_output_dir)
    output_entry.pack(pady=5)

    def browse_output_dir():
        dir_path = filedialog.askdirectory()
        if dir_path:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, dir_path)

    browse_output_btn = tk.Button(root, text="Browse", command=browse_output_dir)
    browse_output_btn.pack(pady=5)

    # Log Display
    log_label = tk.Label(root, text="Logs:")
    log_label.pack(pady=5)

    log_text = tk.Text(root, width=80, height=15, wrap=tk.WORD)
    log_text.pack(pady=5)

    # Extract Button
    extract_btn = tk.Button(root, text="Extract", command=lambda: start_extraction(video_entry.get(), output_entry.get(), log_text))
    extract_btn.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
