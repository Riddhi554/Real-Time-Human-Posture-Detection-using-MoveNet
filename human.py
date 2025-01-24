import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import math

# Load the MoveNet model from TensorFlow Hub
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) -
                         math.atan2(a[1] - b[1], a[0] - b[0]))
    angle = abs(angle)
    return angle if angle <= 180 else 360 - angle

# Function to analyze posture based on detected keypoints
def analyze_posture(keypoints):
    if all(k in keypoints for k in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
        # Calculate angles for sitting, standing, and squatting detection
        left_leg_angle = calculate_angle(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
        right_leg_angle = calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])

        # Posture classification based on angles
        if left_leg_angle > 170 and right_leg_angle > 170:
            return "Standing"
        elif 90 <= left_leg_angle <= 170 and 90 <= right_leg_angle <= 170:
            return "Squatting"
        elif left_leg_angle < 90 and right_leg_angle < 90:
            return "Sitting"
    return "Unknown"

# Function to process the MoveNet output
def process_keypoints(output):
    keypoints = {}
    for idx, keypoint in enumerate(output):
        if keypoint[2] > 0.5:  # Confidence threshold
            keypoints[KEYPOINT_NAMES[idx]] = (keypoint[1], keypoint[0])
    return keypoints

# Keypoint names based on MoveNet's output
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Initialize video capture
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to 192x192 as required by MoveNet
    input_image = cv2.resize(frame, (192, 192))
    input_image = tf.image.resize_with_pad(tf.convert_to_tensor(input_image), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    # Run pose estimation
    outputs = movenet.signatures['serving_default'](input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # Process the detected keypoints
    h, w, _ = frame.shape
    processed_keypoints = process_keypoints(keypoints * [h, w, 1])

    # Draw keypoints on the frame
    for key, coord in processed_keypoints.items():
        cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)

    # Analyze posture based on keypoints
    posture = analyze_posture(processed_keypoints)
    cv2.putText(frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Pose Estimation", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
