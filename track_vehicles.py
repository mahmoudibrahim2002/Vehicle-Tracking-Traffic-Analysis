import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter1d

# Video information and frame generator
SOURCE_VIDEO_PATH = "input_videos/56310-479197605.mp4"
TARGET_VIDEO_PATH = "output_videos/result.mp4"
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Define the source and target coordinates for perspective transformation
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Define lower and upper lines in the source coordinate system
lower_line_source = np.array([[1252, 787],[2298, 803],[-5039, 2159],[-550, 2159]])# Bottom edge of the polygon
upper_line_source = np.array([[-1252, 787],[2298, 803],[-5039, 2159],[-550, 2159]])*2   # Bottom edge of the polygon

## Transform Perspective
class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    
# Smoothing function using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Smoothing function using exponential smoothing
def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]
    for i in range(1, len(data)):
        smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[i - 1])
    return smoothed_data

# Kalman Filter for speed estimation
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0
        self.estimation_error = 1

    def update(self, measurement):
        # Prediction
        self.estimation_error += self.process_variance

        # Update
        kalman_gain = self.estimation_error / (self.estimation_error + self.measurement_variance)
        self.estimated_value += kalman_gain * (measurement - self.estimated_value)
        self.estimation_error *= (1 - kalman_gain)

        return self.estimated_value

# Initialize YOLO model
model = YOLO("yolo11x.pt")

# ByteTrack tracker initialization
CONFIDENCE_THRESHOLD = 0.4
byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD)

# Annotators configuration
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
bounding_box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps, position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

# Polygon zone for detection filtering
polygon_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Transform the lines to the target coordinate system
lower_line_target = view_transformer.transform_points(lower_line_source)
upper_line_target = view_transformer.transform_points(upper_line_source)

# Extract the Y-coordinates of the transformed lines
lower_line_y = lower_line_target[0][1]  # Y-coordinate of the lower line
upper_line_y = upper_line_target[0][1]  # Y-coordinate of the upper line

# Dictionary to store coordinates for each tracked vehicle
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps * 2))  # Larger window for smoothing

# Dictionary to store Kalman filters for each tracker_id
kalman_filters = {}

# Define counting variables
up_count = 0
down_count = 0

# Define traffic level thresholds
TRAFFIC_THRESHOLDS = {"Low": 5, "Moderate": 15}  # Vehicles per minute

def classify_traffic(count):
    if count <= TRAFFIC_THRESHOLDS["Low"]:
        return "Low", (0, 255, 0)  # Green
    elif count <= TRAFFIC_THRESHOLDS["Moderate"]:
        return "Moderate", (0, 255, 255)  # Yellow
    else:
        return "Heavy", (0, 0, 255)  # Red

# Callback function for processing each frame
# Update the counting logic in the callback function
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global up_count, down_count

    # Perform object detection using YOLO
    result = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Filter detections by confidence and class
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]

    # Filter detections outside the polygon zone
    detections = detections[polygon_zone.trigger(detections)]

    # Apply non-max suppression to refine detections
    detections = detections.with_nms(0.5)

    # Update detections with ByteTrack tracker
    detections = byte_track.update_with_detections(detections=detections)

    # Get bottom center coordinates of detections
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    # Transform points to the target ROI
    points = view_transformer.transform_points(points=points).astype(int)

  
    # Store coordinates for each tracked vehicle
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        if tracker_id not in coordinates:
            coordinates[tracker_id] = deque(maxlen=video_info.fps * 2)  # Larger window for smoothing

        prev_y = coordinates[tracker_id][-1] if coordinates[tracker_id] else None
        coordinates[tracker_id].append(y)

        if prev_y is not None:
            # Moving up (crossing the lower line)
            if prev_y > lower_line_y >= y:
                up_count += 1
            # Moving down (crossing the upper line)
            elif prev_y < upper_line_y <= y:
                down_count += 1

    
    # Determine traffic level
    total_count = up_count + down_count
    
    # Count the number of vehicles in the current frame
    current_vehicle_count = len(detections)

    # Determine traffic level based on the current vehicle count
    traffic_level, color = classify_traffic(current_vehicle_count)

    # Overlay vehicle count & traffic level with better visualization
    overlay = frame.copy()
    cv2.rectangle(overlay, (30, 20), (520, 180), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Adding a border around the text area for enhanced clarity
    cv2.rectangle(frame, (30, 20), (520, 180), (255, 255, 255), 2)

    # Display Up&Down vehicle counts and traffic level
    cv2.putText(frame, f"Up: {up_count}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Down: {down_count}", (50, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display Vehicle counts and traffic level
    cv2.putText(frame, f"Vehicles: {current_vehicle_count}", (50, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Traffic Intensity: {traffic_level}", (50, 170), cv2.FONT_HERSHEY_DUPLEX, 1, color, 3, cv2.LINE_AA)

    # Format labels for detections
    labels = []
    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # Calculate speed using multiple points
            y_coords = list(coordinates[tracker_id])
            distances = np.abs(np.diff(y_coords))
            times = np.arange(len(y_coords)) / video_info.fps
            speeds = distances / np.diff(times) * 3.6  # Convert to km/h

            # Apply smoothing (e.g., moving average)
            smoothed_speeds = moving_average(speeds, window_size=5)

            # Initialize Kalman filter for new tracker_id
            if tracker_id not in kalman_filters:
                kalman_filters[tracker_id] = KalmanFilter(process_variance=1e-5, measurement_variance=0.1)

            # Apply Kalman filter to smooth the speed
            filtered_speed = kalman_filters[tracker_id].update(np.mean(smoothed_speeds))

            # Append the filtered speed to the labels
            labels.append(f"#{tracker_id} {int(filtered_speed)} km/h")

    # Annotate the frame with bounding boxes, traces, and labels
    annotated_frame = bounding_box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return annotated_frame

# Process the video with a progress bar
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video"):
        annotated_frame = callback(frame, 0)
        sink.write_frame(annotated_frame)
