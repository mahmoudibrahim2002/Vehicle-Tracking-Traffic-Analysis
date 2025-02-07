# Vehicle Tracking &amp; Traffic Analysis | Computer Vision Project (Speed and Traffic intensity Estimation)

## 🌐 Overview
This project implements an advanced vehicle tracking and speed estimation system using **YOLO for object detection**, **ByteTrack for tracking**, and the **Supervision library** for annotation and visualization. It also includes additional features such as vehicle counting, traffic intensity classification, and smoothing techniques for stable speed estimation.

## Features
- **Vehicle Detection:** Uses YOLO for real-time object detection.
- **Object Tracking:** ByteTrack ensures smooth tracking across frames.
- **Speed Estimation:** Calculates vehicle speed based on positional changes over time.
- **Noise Reduction:** Implements Kalman filters and smoothing techniques to stabilize speed calculations.
- **Traffic Analysis:** Tracks the number of vehicles moving up/down and classifies traffic intensity.
- **Visualization:** Displays vehicle count, speed, and traffic status in a user-friendly format.

## Project Stages
### 1. Initial Implementation (Roboflow Tutorial-Based)
- Implemented vehicle tracking based on [Supervision's Roboflow tutorial](https://supervision.roboflow.com/how_to/track_objects/).
- Estimated speed using the first and last position of vehicles in a deque.
- **Issue:** Inaccurate results due to overlapping bounding boxes when multiple vehicles appeared in the same frame.

### 2. Improved Speed Detection
- Enhanced speed stability using:
  - **Smoothing functions:** Moving average and exponential smoothing.
  - **Kalman Filter:** Reduces noise and fluctuations in speed estimation.
  - **Increased deque size:** Stores more frames for better averaging.
  - **Multiple points for speed calculation:** Provides more reliable estimations.
- **Result:** More consistent and stable speed display.

### 3. Vehicle Counting and Traffic Intensity Analysis
- Implemented vehicle counting logic for detecting vehicles moving **up** and **down**.
- Displayed real-time traffic intensity using threshold-based classification:
  - **Low Traffic:** Green (≤ 5 vehicles/min)
  - **Moderate Traffic:** Yellow (≤ 15 vehicles/min)
  - **Heavy Traffic:** Red (> 15 vehicles/min)

## How It Works
1. **YOLO detects vehicles** in each frame.
2. **ByteTrack assigns tracking IDs** to detected vehicles.
3. **Position data is stored** for each vehicle.
4. **Speed is calculated** using the change in position over time.
5. **Speed smoothing is applied** (moving average, exponential smoothing, and Kalman filter).
6. **Vehicle movement is analyzed**, counting those moving up or down.
7. **Traffic intensity is classified** and displayed on the video output.

## 📁 File Descriptions

- 📓 **`Traffic Red-Light Running Violation Detection.ipynb`**: The primary Jupyter notebook containing all code and explanations for this project.
- 🎥 **`traffic_video.mp4`**: Sample video footage used to detect traffic violations.
- 📄 **`haarcascade_russian_plate_number.xml`**: XML file used for license plate detection.
- 📘 **`README.md`**: You're currently reading this file! Provides an overview and useful information about the project.

## 🚀 Instructions for Local Execution

1. **Clone this Repository**: First and foremost, clone this repo to your local machine.
2. **Open the Notebook**: Launch the `Traffic Red-Light Running Violation Detection.ipynb` in Jupyter.
3. **Setup Dependencies**: Make sure you've installed all necessary Python libraries and have a local MySQL database running.
4. **Database Credentials**: Inside the notebook, replace the `your_username` and `your_password` placeholders in the database connection section with your actual database credentials.
5. **Execution**: Execute all cells in the notebook to view the results.

## Installation
```bash
pip install ultralytics supervision opencv-python numpy scipy tqdm
```

## Usage
```python
python track_vehicles.py
```

## Example Output
![Vehicle Tracking Demo](https://img.youtube.com/vi/dzHYjDuRYzs/0.jpg)](https://youtu.be/dzHYjDuRYzs)

---
- 🤝 **Connect on LinkedIn**: [LinkedIn](mahmoud-ibrahim2002)
- 🌐 **Kaggle Notebook**: Interested in a Kaggle environment? Explore the notebook [here](https://www.kaggle.com/code/farzadnekouei/traffic-red-light-running-violation-detection).
- 📹 **Input Video Data**: Access the raw and modified video [here](https://www.kaggle.com/datasets/farzadnekouei/license-plate-recognition-for-red-light-violation).
- 🎥 **Project Demo**: Watch the live demonstration of this project on [YouTube](https://www.youtube.com/watch?v=dzHYjDuRYzs).
