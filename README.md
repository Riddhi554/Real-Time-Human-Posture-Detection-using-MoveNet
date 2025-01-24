# Real-Time Human Posture Detection using MoveNet 🚶‍♂️🧘‍♀️

This project implements a **real-time human posture detection system** using TensorFlow's **MoveNet SinglePose Thunder** model. It processes video feed from a webcam, performs pose estimation, and classifies the posture into predefined categories: **Standing**, **Sitting**, or **Squatting**. 

---

## Features ✨

- **Pose Estimation**: Detects 17 keypoints of the human body (e.g., nose, shoulders, hips, knees, ankles, etc.). 🦵
- **Posture Classification**: Classifies posture based on calculated joint angles:
  - **Standing** 🚶‍♂️
  - **Sitting** 🧘‍♀️
  - **Squatting** 🏋️‍♀️
  - **Unknown** 🤷‍♂️ (if detection confidence is low or insufficient keypoints are detected).
- **Real-Time Performance**: Processes video frames from a webcam in real-time. 🎥
- **Visual Feedback**: Displays detected keypoints and posture classification on the video feed. 👀

---

## How It Works ⚙️

1. **Pose Estimation**:
   - The MoveNet model processes 192x192 input images. 🖼️
   - Outputs normalized keypoints with x, y coordinates and a confidence score. 📊

2. **Posture Classification**:
   - Angles between joints are calculated using trigonometric functions. 📐
   - Classification is based on predefined thresholds:
     - **Standing**: Both leg angles > 170°.
     - **Sitting**: Both leg angles < 90°.
     - **Squatting**: Leg angles between 90° and 170°.

3. **Real-Time Visualization**:
   - Draws keypoints on the video feed. 🎨
   - Displays the detected posture as text on the screen. 📝

---

## Technologies Used 🛠️

- **TensorFlow**: For loading and running the MoveNet model. 🤖
- **TensorFlow Hub**: To download and use the MoveNet SinglePose Thunder model. 🧠
- **OpenCV**: For video capture, frame resizing, and visualization. 💻
- **NumPy**: For mathematical operations and angle calculations. 📐
- **Math**: For trigonometric functions. 🔢

---

## Installation 🧑‍💻

### Prerequisites 📋

- Python 3.8 or higher
- Webcam for real-time video input 📸

### Steps 📥

1. Clone the repository:
   ```bash
   git clone https://github.com/Riddhi554/real-time-posture-detection.git
   cd real-time-posture-detection
