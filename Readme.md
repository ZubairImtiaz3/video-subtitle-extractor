
# Video Subtitle Extractor

It is a Python tool that allows you to extract subtitles from videos by taking screenshots at one-second intervals, performing OCR (Optical Character Recognition) on the screenshots, and saving the detected subtitles to a text file. It provides a graphical user interface (GUI) using Tkinter for ease of use.

## Features

- Extracts frames from video files at 1-second intervals.
- Applies OCR using EasyOCR to extract text from images.
- Identifies subtitle regions in video frames and extracts text.
- Cleans and filters subtitle text for accuracy.
- Saves the extracted subtitles to a `.txt` file.
- Provides a simple GUI for video file and output directory selection.
- Keeps the GUI responsive by running extraction in a separate thread.

## Requirements

- Python 3.x
- OpenCV
- EasyOCR
- Pillow
- Tkinter (should be included with Python)
- NumPy

## Installation

To get started with Subtitle Extractor, clone this repository and install the required dependencies.

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/subtitle-extractor.git
    ```

2. Navigate to the project directory:

    ```bash
    cd subtitle-extractor
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:

    ```bash
    python subtitle_extractor.py
    ```

2. The GUI will open. Use the following features:
   - **Select Video:** Choose the video file you want to extract subtitles from.
   - **Select Output Directory:** Choose where the extracted subtitles and screenshots will be saved.
   - **Extract:** Click this button to start the extraction process.

3. The process will extract screenshots, perform OCR to identify subtitles, and save them in a text file named `output_subtitles.txt`.

## GUI Layout

- **Video Path Selection:** Choose a video file (e.g., `.mp4`, `.avi`, `.mkv`, `.mov`).
- **Output Directory Selection:** Choose the folder where the screenshots and subtitle text file will be saved.
- **Log Display:** View the progress and status messages of the extraction process.
- **Extract Button:** Start the extraction process after selecting the video and output directory.
